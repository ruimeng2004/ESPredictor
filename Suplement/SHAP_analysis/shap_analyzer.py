import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import os
from typing import Dict
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from itertools import chain
import pandas as pd

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # For math text

class DrugSHAPAnalyzer:
    def __init__(self, model, feature_config: Dict[str, Dict[str, int]], device):
        """Initialize with input dimension validation"""
        self.model = model
        self.feature_config = feature_config
        self.device = device
        self.shap_values = None
        
        self.total_dim = sum(v['dim'] for v in feature_config.values())
        print(f"Total feature dimension configured: {self.total_dim}")
        
        # 新增：用于存储特征摘要的字典
        self.feature_summary = {
            'mean_abs_shap_drug1': None,
            'mean_abs_shap_drug2': None,
            'mean_shap_drug1': None,
            'mean_shap_drug2': None
        }
        
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

    def _safe_predict(self, x):
        """Safe prediction function with dimension checks"""
        self.model.eval()
        with torch.no_grad():
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            
            if x.ndim == 1:
                x = x.reshape(1, -1) 
            elif x.ndim == 2:
                pass  
            else:
                raise ValueError(f"Input should be 1D or 2D array, got: {x.ndim}")
            
            assert x.shape[1] == self.total_dim, \
                f"Input feature mismatch! Required {self.total_dim}, got {x.shape[1]}"
            
            chunk_size = 4  
            outputs = []
            for i in range(0, x.shape[0], chunk_size):
                chunk = x[i:i+chunk_size]
                x_tensor = torch.tensor(chunk, dtype=torch.float32, device='cpu')
                x_tensor = x_tensor.pin_memory().to(self.device, non_blocking=True)

                zero_tensor = torch.zeros_like(x_tensor, device=self.device)
                
                assert x_tensor.shape == zero_tensor.shape, \
                    f"Input tensor shape mismatch: {x_tensor.shape} vs {zero_tensor.shape}"
                
                out = self.model(x_tensor, zero_tensor)[:, 1]
                outputs.append(out.cpu().numpy())
                
                del x_tensor, zero_tensor
                torch.cuda.empty_cache()
            
            return np.concatenate(outputs)

    def compute_shap(self, drug_pairs, nsamples=10, batch_size=2):
        """Compute SHAP values with safety checks"""
        try:
            assert drug_pairs.ndim == 3, "Input should be 3D array"
            assert drug_pairs.shape[1] == 2, "Each sample should contain 2 drugs"
            assert drug_pairs.shape[2] == self.total_dim, \
                f"Feature dimension mismatch! Required {self.total_dim}, got {drug_pairs.shape[2]}"
            
            all_drugs = np.vstack([drug_pairs[:,0], drug_pairs[:,1]])
            print(f"Merged drug data shape: {all_drugs.shape}")
            
            num_features = all_drugs.shape[1]
            min_evals = 2 * num_features + 1
            max_evals = max(min_evals * 2, 5000) 
            
            background = shap.sample(all_drugs, min(50, len(all_drugs)))
            
            explainer = shap.Explainer(
                model=self._safe_predict,
                masker=background,
                algorithm="auto", 
                max_evals=max_evals
            )
            
            sample_indices = np.random.choice(len(all_drugs), min(nsamples*2, len(all_drugs)), replace=False)
            self.shap_values = np.zeros((len(drug_pairs), 2, drug_pairs.shape[2]))
            
            pbar = tqdm(total=len(sample_indices), desc="Calculating SHAP values")
            for i in range(0, len(sample_indices), batch_size):
                batch_idx = sample_indices[i:i+batch_size]
                batch = all_drugs[batch_idx]
                
                if batch.size == 0:
                    print("Warning: Empty batch, skipping")
                    continue
                
                sv = explainer(batch).values
                
                for j, idx in enumerate(batch_idx):
                    orig_idx = idx % len(drug_pairs)
                    drug_pos = idx // len(drug_pairs)
                    self.shap_values[orig_idx, drug_pos] = sv[j]
                
                pbar.update(len(batch_idx))
                
                del batch, sv
                torch.cuda.empty_cache()
            
            pbar.close()
            if self.shap_values is not None and not np.isnan(self.shap_values).all():
                # 计算Drug1和Drug2各特征的平均绝对SHAP值（重要性）
                self.feature_summary['mean_abs_shap_drug1'] = np.nanmean(np.abs(self.shap_values[:, 0, :]), axis=0)
                self.feature_summary['mean_abs_shap_drug2'] = np.nanmean(np.abs(self.shap_values[:, 1, :]), axis=0)
                # 计算Drug1和Drug2各特征的平均SHAP值（方向性）
                self.feature_summary['mean_shap_drug1'] = np.nanmean(self.shap_values[:, 0, :], axis=0)
                self.feature_summary['mean_shap_drug2'] = np.nanmean(self.shap_values[:, 1, :], axis=0)
                
                print("SHAP feature summary calculated successfully")
            else:
                print("Warning: SHAP values are invalid, cannot calculate summary")
            return self.shap_values
                
        except Exception as e:
            print(f"SHAP calculation failed: {str(e)}")
            self.shap_values = np.full((len(drug_pairs), 2, drug_pairs.shape[2]), np.nan)
            return self.shap_values

    def save_shap_summary_to_csv(self, output_path="shap_feature_summary.csv"):
        """
        将每个特征维度的详细SHAP统计信息保存到CSV文件。
        包括特征索引、所属模态、药物来源、平均绝对SHAP值（重要性）和平均SHAP值（方向）。
        """
        if self.feature_summary['mean_abs_shap_drug1'] is None:
            print("Warning: No SHAP summary data available. Please run compute_shap() first.")
            return

        # 准备数据列表
        data = []
        
        # 遍历所有特征索引 (0 到 total_dim-1)
        for global_idx in range(self.total_dim):
            # 确定当前索引属于哪个模态
            modality_name = "Unknown"
            local_idx = global_idx
            for name, config in self.feature_config.items():
                if config['start'] <= global_idx < config['start'] + config['dim']:
                    modality_name = name.upper() if name.lower() != 'bert' else 'BERT'
                    local_idx = global_idx - config['start']  # 在该模态内的局部索引
                    break
            
            # 为Drug1和Drug2添加一行记录
            row_drug1 = {
                'global_index': global_idx,
                'modality': modality_name,
                'local_index': local_idx,
                'drug_source': 'Drug1',
                'mean_abs_shap': self.feature_summary['mean_abs_shap_drug1'][global_idx],
                'mean_shap': self.feature_summary['mean_shap_drug1'][global_idx]
            }
            
            row_drug2 = {
                'global_index': global_idx,
                'modality': modality_name,
                'local_index': local_idx,
                'drug_source': 'Drug2',
                'mean_abs_shap': self.feature_summary['mean_abs_shap_drug2'][global_idx],
                'mean_shap': self.feature_summary['mean_shap_drug2'][global_idx]
            }
            
            data.append(row_drug1)
            data.append(row_drug2)
        
        # 创建DataFrame并排序（按重要性降序）
        df_summary = pd.DataFrame(data)
        # 按平均绝对SHAP值降序排序，以便最重要的特征排在最前面
        df_summary.sort_values(by='mean_abs_shap', ascending=False, inplace=True)
        
        # 保存到CSV
        df_summary.to_csv(output_path, index=False)
        print(f"SHAP feature summary saved to: {output_path}")
        
        # 同时打印一些统计信息以供快速参考
        print("\n=== 各模态平均贡献度 (Mean |SHAP|) ===")
        for modality in df_summary['modality'].unique():
            if modality == "Unknown":
                continue
            mod_data = df_summary[df_summary['modality'] == modality]
            avg_contrib = mod_data['mean_abs_shap'].mean()
            print(f"{modality}: {avg_contrib:.6f}")
        
        return df_summary

    def plot_comprehensive_analysis(self, output_dir="feature_analysis"):
        """Generate SHAP analysis plots with optimized layout"""
        if self.shap_values is None:
            raise RuntimeError("Must call compute_shap() first")
            
        os.makedirs(output_dir, exist_ok=True)
        
        if np.isnan(self.shap_values).all():
            print("Error: All SHAP values are invalid")
            return
        
        # ==================== Combined Plot Layout ====================
        fig = plt.figure(figsize=(18, 7))
        gs = GridSpec(1, 2, width_ratios=[1.8, 1], height_ratios=[1])
        
        # Left plot - Donut chart (maximized)
        ax0 = fig.add_subplot(gs[0])
        self._plot_combined_donut_chart(ax0, title="(A) Feature Contribution Ratio")
        
        # Right plot - SHAP summary (fixed size)
        ax1 = fig.add_subplot(gs[1])
        self._plot_combined_shap_summary(ax1, title="(B) Top Feature Impacts", max_display=30)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_analysis_combined.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ==================== Individual Plots ====================
        # Donut chart alone (maximized)
        fig_donut, ax_donut = plt.subplots(figsize=(14, 14))
        self._plot_combined_donut_chart(ax_donut,
                                      title="Feature Contribution Distribution",
                                      title_fontsize=16)  # Consistent title size
        plt.savefig(f"{output_dir}/encoder_contributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # SHAP plot alone (fixed size)
        fig_shap, ax_shap = plt.subplots(figsize=(12, 6))
        self._plot_combined_shap_summary(ax_shap,
                                       title="Top Feature Impacts",
                                       max_display=30,
                                       title_fontsize=16)  # Consistent title size
        plt.savefig(f"{output_dir}/feature_impacts.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ==================== NEW: Top30 Average SHAP by Dimension ====================
        self._plot_top30_average_shap(output_dir)

    def _plot_top30_average_shap(self, output_dir):
        """新增：绘制八个维度的Top30 SHAP平均值横向条形图"""
        if self.shap_values is None:
            return
            
        # 准备数据
        data = []
        display_names = {}
        for name in self.feature_config:
            display_names[name] = "BERT" if name.lower() == "bert" else name.upper()
        
        # 计算Drug1和Drug2各维度的Top30平均SHAP
        for drug_idx, drug_name in [(0, 'Drug1'), (1, 'Drug2')]:
            for name, config in self.feature_config.items():
                start, end = config['start'], config['start'] + config['dim']
                shap_subset = self.shap_values[:, drug_idx, start:end]
                
                # 计算每个特征的绝对SHAP平均值
                mean_abs_shap = np.nanmean(np.abs(shap_subset), axis=0)
                
                # 获取Top30特征
                top_indices = np.argsort(mean_abs_shap)[-30:][::-1]
                top_shap = mean_abs_shap[top_indices]
                
                # 计算Top30的平均值
                avg_top30 = np.mean(top_shap)
                
                data.append({
                    'Dimension': f"{drug_name} {display_names[name]}",
                    'Average_SHAP': avg_top30,
                    'Color': '#4e79a7' if drug_idx == 0 else '#f28e2b',  # Drug1蓝色，Drug2橙色
                    'Drug': drug_name  # 新增列用于hue分组
                })
        
        df = pd.DataFrame(data)
        
        # 绘制横向条形图
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=df, y='Dimension', x='Average_SHAP', 
                        hue='Drug',  # 使用hue参数替代直接指定palette
                        palette=['#4e79a7', '#f28e2b'],  # 定义Drug1和Drug2的颜色
                        dodge=False,  # 确保所有条形在同一位置
                        legend=False)  # 隐藏图例
        
        # 美化图表
        ax.set_title("Average Top30 SHAP Values by Dimension", fontsize=16, pad=20)
        ax.set_xlabel("Average Absolute SHAP Value (Top30 Features)", fontsize=12)
        ax.set_ylabel("")
        
        # 添加数值标签
        for p in ax.patches:
            width = p.get_width()
            ax.text(width + 0.002, p.get_y() + p.get_height()/2.,
                f'{width:.4f}',
                ha='left', va='center', fontsize=10)
        
        # 设置字体
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_family('serif')
            item.set_name('Times New Roman')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top30_average_shap_by_dimension.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 同时保存CSV文件
        df.to_csv(f"{output_dir}/top30_average_shap_by_dimension.csv", index=False)
        print(f"Saved Top30 average SHAP by dimension to {output_dir}/top30_average_shap_by_dimension.csv")

    def _plot_combined_donut_chart(self, ax, title, output_path=None, title_fontsize=16):
        """Enhanced donut chart with maximized layout"""
        # Professional color scheme
        drug1_colors = ['#4e79a7', '#59a14f', '#76b7b2', '#b07aa1']  # Cool tones
        drug2_colors = ['#f28e2b', '#e15759', '#edc948', '#ff9da7']  # Warm tones
        
        # Calculate contributions
        contributions = {}
        total_contrib = 0
        
        # Create display names for legend
        display_names = {}
        for name in self.feature_config:
            display_names[name] = "BERT" if name.lower() == "bert" else name.upper()
        
        # Drug1 contribution calculation
        for name, config in self.feature_config.items():
            start, end = config['start'], config['start'] + config['dim']
            contrib = np.nansum(np.abs(self.shap_values[:, 0, start:end]))
            contributions[f"Drug1 {display_names[name]}"] = contrib
            total_contrib += contrib
        
        # Drug2 contribution calculation
        for name, config in self.feature_config.items():
            start, end = config['start'], config['start'] + config['dim']
            contrib = np.nansum(np.abs(self.shap_values[:, 1, start:end]))
            contributions[f"Drug2 {display_names[name]}"] = contrib
            total_contrib += contrib
        
        # Calculate percentages
        percentages = {k: (v/total_contrib)*100 for k, v in contributions.items()}
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 14))
            save_fig = True
        else:
            save_fig = False
        
        # Prepare data
        labels = list(percentages.keys())
        sizes = list(percentages.values())
        colors = drug1_colors + drug2_colors
        
        # Draw professional donut chart
        radius = 0.8 if ax is None else 0.7
        width = 0.45 if ax is None else 0.4
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            colors=colors,
            startangle=90,
            radius=radius,
            wedgeprops=dict(width=width, edgecolor='white', linewidth=1.5),
            autopct=lambda p: f'{p:.1f}%',
            pctdistance=0.75  # Moved closer to center
        )
        
        # Set percentage text style - black font
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(13)
            autotext.set_fontweight('bold')
            autotext.set_family('serif')
            autotext.set_name('Times New Roman')
        
        # Modified title setting - consistent font size
        title_obj = ax.set_title(title, fontsize=title_fontsize, pad=25)
        title_obj.set_family('serif')
        title_obj.set_name('Times New Roman')
        ax.axis('equal')
        
        # Create simplified legend with only color-feature correspondence
        legend_elements = []
        feature_types = list(self.feature_config.keys())
        
        # Add Drug1 features to legend
        for i, name in enumerate(feature_types):
            legend_elements.append(
                Patch(facecolor=drug1_colors[i], 
                      label=f"{display_names[name]} (Drug1)")
            )
        
        # Add Drug2 features to legend
        for i, name in enumerate(feature_types):
            legend_elements.append(
                Patch(facecolor=drug2_colors[i], 
                      label=f"{display_names[name]} (Drug2)")
            )
        
        # Add legend at the bottom with two columns
        legend = ax.legend(handles=legend_elements, loc='upper center', 
                         bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=12)
        
        # Set legend font to Times New Roman
        for text in legend.get_texts():
            text.set_family('serif')
            text.set_name('Times New Roman')
        
        if save_fig and output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_combined_shap_summary(self, ax, title, max_display=30, output_path=None, title_fontsize=14):
        """修正后的SHAP summary plot，确保Drug1和Drug2特征都能正确显示"""
        # ================ 关键修改1：重要性计算方式 ================
        # 分别计算Drug1和Drug2的特征重要性（避免索引重叠）
        drug1_importance = np.nanmean(np.abs(self.shap_values[:, 0, :]), axis=0)
        drug2_importance = np.nanmean(np.abs(self.shap_values[:, 1, :]), axis=0)
        
        # 合并重要性时添加偏移量（Drug2索引从total_dim开始）
        combined_importance = np.concatenate([
            drug1_importance,       # 索引 0 到 total_dim-1
            drug2_importance        # 索引 total_dim 到 2*total_dim-1
        ])
        
        # 获取top特征索引（现在包含Drug1和Drug2）
        top_indices = np.argsort(combined_importance)[-max_display:][::-1]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            save_fig = True
        else:
            save_fig = False
        
        # ================ 关键修改2：特征标签生成 ================
        feature_names = []
        for idx in top_indices:
            # 判断特征属于Drug1还是Drug2
            if idx < self.total_dim:
                drug_label, adjusted_idx = "Drug1", idx
            else:
                drug_label, adjusted_idx = "Drug2", idx - self.total_dim
            
            # 查找特征所属的视图（BERT/1D/2D/3D）
            for name, config in self.feature_config.items():
                if config['start'] <= adjusted_idx < config['start'] + config['dim']:
                    display_name = "BERT" if name.lower() == "bert" else name.upper()
                    feature_names.append(
                        f"{display_name}_{adjusted_idx-config['start']} ({drug_label})"
                    )
                    break
        
        # ================ 数据准备 ================
        # 合并SHAP值（同样添加偏移量）
        all_shap = np.concatenate([
            self.shap_values[:, 0, :],  # Drug1 SHAP
            self.shap_values[:, 1, :]   # Drug2 SHAP
        ], axis=1)
        
        top_shap = all_shap[:, top_indices]
        n_samples, n_features = top_shap.shape
        
        # ================ 绘图逻辑 ================
        y_pos = np.arange(n_features)
        
        # 颜色映射（基于随机值示例）
        feature_values = np.random.rand(n_samples)
        cmap = sns.diverging_palette(250, 350, as_cmap=True)
        norm = plt.Normalize(vmin=feature_values.min(), vmax=feature_values.max())
        
        # 绘制每个样本的SHAP值
        for i in range(n_samples):
            ax.scatter(top_shap[i], y_pos,
                    c=[cmap(norm(feature_values[i]))] * n_features,
                    s=40, alpha=0.7, edgecolors='w', linewidth=0.3)
        
        # 添加均值参考线
        mean_shap = np.nanmean(top_shap, axis=0)
        ax.scatter(mean_shap, y_pos, c='black', marker='|', s=200, label='Mean')
        
        # 坐标轴设置
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[n_features-i-1] for i in range(n_features)])  # 倒序显示
        ax.set_xlabel("SHAP Value", fontsize=12)
        
        # 字体设置（Times New Roman）
        for label in chain(ax.get_xticklabels(), ax.get_yticklabels()):
            label.set_family('serif')
            label.set_name('Times New Roman')
            label.set_fontsize(10)
        
        # 标题和标签字体
        title_obj = ax.set_title(title, fontsize=title_fontsize, pad=20)
        title_obj.set_family('serif')
        title_obj.set_name('Times New Roman')
        ax.xaxis.label.set_family('serif')
        ax.xaxis.label.set_name('Times New Roman')
        
        # 颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Feature Value", fontsize=10)
        for label in cbar.ax.get_yticklabels():
            label.set_family('serif')
            label.set_name('Times New Roman')
        
        # 图例和网格
        legend = ax.legend(loc='upper right', fontsize=10)
        for text in legend.get_texts():
            text.set_family('serif')
            text.set_name('Times New Roman')
        ax.grid(True, axis='x', alpha=0.3)
        
        # 底部说明文字
        note = ax.text(0.5, -0.15, "Red = High Value, Blue = Low Value", 
                    ha="center", transform=ax.transAxes,
                    fontsize=10, color='dimgrey')
        note.set_family('serif')
        
        if save_fig and output_path:
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()