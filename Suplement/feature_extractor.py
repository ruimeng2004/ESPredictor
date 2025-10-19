import numpy as np
import torch
import pandas as pd
import os
import traceback
from rdkit import Chem
from rdkit.Chem import AllChem
from Suplement.FE.FE_BERT.FE_Bert import gnnsmile
from Suplement.FE.FE_3Ds.preprocess import smiles_to_data, prepare_batch
from Suplement.FE.FE_3Ds.pretrain_MolMVC import EmbeddingModel

def process_dataset(train_data, feature_type, device, save_dir="extracted_features"):
    """
    处理药物特征并存储特征数据
    
    参数:
        train_data: 包含药物ID和SMILES的数据框
        feature_type: 特征类型 ('bert', '1D', '2D', '3D', 'multi', 'fingerprint')
        device: 计算设备 (CPU/GPU)
        save_dir: 特征存储目录
        
    返回:
        alldrugtrain: 包含药物特征向量的字典
        ftrain: 有效药物ID列表
    """
    os.makedirs(save_dir, exist_ok=True)
    
    drug1_ids = np.array(train_data["Drug1_ID"])
    drug2_ids = np.array(train_data["Drug2_ID"])
    all_drugs = np.unique(np.append(drug1_ids, drug2_ids))
    ftrain = list(all_drugs)
    
    alldrugtrain = {}
    removelist = []
    
    print(f"开始处理 {len(ftrain)} 个药物...")
    
    for drug_id in ftrain:
        try:
            # 记录当前处理的药物ID
            print(f"\n处理药物: {drug_id}")
            
            # 尝试在Drug1_ID列中找到药物
            try:
                index = list(drug1_ids).index(drug_id)
                drug_col = "Drug1_ID"
                smiles_col = train_data.columns[1] 
                print(f"在Drug1_ID列中找到药物，索引: {index}")
            except ValueError:
                # 如果在Drug1_ID中找不到，尝试在Drug2_ID中找
                try:
                    index = list(drug2_ids).index(drug_id)
                    drug_col = "Drug2_ID"
                    smiles_col = train_data.columns[3]
                    print(f"在Drug2_ID列中找到药物，索引: {index}")
                except ValueError:
                    removelist.append(drug_id)
                    print(f"错误: 无法在数据集中找到药物 {drug_id}")
                    continue
            
            # 获取药物名称和SMILES
            drug_name = train_data.iloc[index][drug_col]
            smiles = train_data.iloc[index][smiles_col]
            print(f"药物名称: {drug_name}, SMILES: {smiles}")
            
            # 特征提取
            if feature_type == 'bert':
                print("使用BERT特征提取...")
                drug_fingerprints = gnnsmile(smiles)
                drug_fingerprints = list(drug_fingerprints[0])
                print(f"BERT特征维度: {len(drug_fingerprints)}")
                
            elif feature_type in ['1D', '2D', '3D', 'multi']:
                print(f"使用MolMVC模型提取{feature_type}特征...")
                
                # 步骤1: 将SMILES转换为数据对象
                print("步骤1: 将SMILES转换为数据对象...")
                try:
                    data = smiles_to_data(smiles, xyz_path=None)
                    print("数据对象创建成功")
                except Exception as e:
                    print(f"错误: 无法将SMILES转换为数据对象: {str(e)}")
                    removelist.append(drug_id)
                    continue
                
                # 步骤2: 准备批次数据
                print("步骤2: 准备批次数据...")
                try:
                    batch = prepare_batch(data)
                    print("批次数据准备成功")
                except Exception as e:
                    print(f"错误: 无法准备批次数据: {str(e)}")
                    removelist.append(drug_id)
                    continue
                
                # 步骤3: 加载模型
                print("步骤3: 加载EmbeddingModel...")
                try:
                    model = EmbeddingModel().to(device)
                    model.eval()
                    print("模型加载成功")
                except Exception as e:
                    print(f"错误: 无法加载模型: {str(e)}")
                    removelist.append(drug_id)
                    continue
                
                # 步骤4: 提取嵌入
                print("步骤4: 提取嵌入...")
                try:
                    with torch.no_grad():
                        emb_1d_low, emb_1d_high, emb_2d, emb_3d = model(batch)
                    print("嵌入提取成功")
                except Exception as e:
                    print(f"错误: 提取嵌入失败: {str(e)}")
                    removelist.append(drug_id)
                    continue
                
                # 根据特征类型处理嵌入
                if feature_type == "1D":
                    print("处理1D特征...")
                    emb_1d = (emb_1d_low + emb_1d_high) / 2
                    drug_fingerprints = emb_1d.cpu().numpy().flatten().tolist()
                    print(f"1D特征维度: {len(drug_fingerprints)}")
                    
                elif feature_type == "2D":
                    print("处理2D特征...")
                    drug_fingerprints = emb_2d.cpu().numpy().flatten().tolist()
                    print(f"2D特征维度: {len(drug_fingerprints)}")
                    
                elif feature_type == "3D":
                    print("处理3D特征...")
                    drug_fingerprints = emb_3d.cpu().numpy().flatten().tolist()
                    print(f"3D特征维度: {len(drug_fingerprints)}")
                    
                elif feature_type == "multi":
                    print("处理多模态特征...")
                    emb_1d = (emb_1d_low + emb_1d_high) / 2
                    
                    # 提取BERT特征
                    print("提取BERT特征...")
                    try:
                        bert_features = np.array(gnnsmile(smiles)[0])
                        print(f"BERT特征维度: {len(bert_features)}")
                    except Exception as e:
                        print(f"错误: 提取BERT特征失败: {str(e)}")
                        removelist.append(drug_id)
                        continue
                    
                    # 组合所有特征
                    print("组合所有特征...")
                    try:
                        drug_fingerprints = np.concatenate([
                            emb_1d.cpu().numpy().flatten(),
                            emb_2d.cpu().numpy().flatten(),
                            emb_3d.cpu().numpy().flatten(),
                            bert_features
                        ]).tolist()
                        print(f"多模态特征总维度: {len(drug_fingerprints)}")
                    except Exception as e:
                        print(f"错误: 组合特征失败: {str(e)}")
                        removelist.append(drug_id)
                        continue
                
            elif feature_type == "fingerprint":
                print("使用指纹特征提取...")
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        print(f"错误: 无法从SMILES创建分子: {smiles}")
                        removelist.append(drug_id)
                        continue
                    
                    fps = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
                    drug_fingerprints = [float(bit) for bit in fps.ToBitString()]
                    print(f"指纹特征维度: {len(drug_fingerprints)}")
                except Exception as e:
                    print(f"错误: 提取指纹特征失败: {str(e)}")
                    removelist.append(drug_id)
                    continue
                
            else:
                raise ValueError(f"未知的特征类型: {feature_type}")
                
            # 将特征添加到字典 - 使用原始drug_id作为键
            alldrugtrain[drug_id] = drug_fingerprints
            print(f"成功为药物 {drug_id} 提取特征")
            
        except Exception as e:
            print(f"处理药物 {drug_id} 时发生未捕获的异常: {str(e)}")
            traceback.print_exc()
            removelist.append(drug_id)
    
    # 从ftrain中移除处理失败的药物
    for drug_id in removelist:
        if drug_id in ftrain:
            ftrain.remove(drug_id)
    
    # 保存特征数据
    if alldrugtrain:
        # 转换为特征矩阵 (n_samples, n_features)
        feature_matrix = np.array(list(alldrugtrain.values()))
        
        # 存储特征数据
        np.save(f"{save_dir}/{feature_type}_features.npy", feature_matrix)
        
        # 存储药物ID列表
        np.save(f"{save_dir}/{feature_type}_drug_ids.npy", np.array(ftrain))
        
        # 打印维度信息
        print(f"\n=== {feature_type}特征存储完成 ===")
        print(f"药物数量: {len(alldrugtrain)}")
        print(f"特征维度: {feature_matrix.shape[1]}")
        print(f"总样本数: {feature_matrix.shape[0]}")
        print(f"存储路径: {save_dir}/{feature_type}_features.npy")
        
        # 检查数据有效性
        if np.isnan(feature_matrix).any():
            print("警告: 特征数据包含NaN值!")
        if np.isinf(feature_matrix).any():
            print("警告: 特征数据包含Inf值!")
        
    else:
        print("警告: 没有成功提取任何药物特征!")
    
    # 打印处理结果统计
    print(f"\n处理结果统计:")
    print(f"总药物数量: {len(ftrain) + len(removelist)}")
    print(f"成功处理: {len(alldrugtrain)}")
    print(f"处理失败: {len(removelist)}")
    
    # 保存失败药物列表
    if removelist:
        pd.DataFrame({'failed_drugs': removelist}).to_csv(f"{save_dir}/failed_drugs.csv", index=False)
        print(f"保存失败药物列表到 {save_dir}/failed_drugs.csv")
    
    return alldrugtrain, ftrain