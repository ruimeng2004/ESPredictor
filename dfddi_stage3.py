import time
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.model_selection import KFold
import torch.nn.functional as F
from Suplement.config import get_args
from Suplement.feature_extractor import process_dataset
from Suplement.Network.DrugInteractionModel import DrugInteractionModel
from Suplement.SHAP_analysis.shap_analyzer import DrugSHAPAnalyzer
import matplotlib.pyplot as plt

# ===================== Model Training Function =====================
def train_model(model, train_loader, optimizer, criterion, margin_criterion, device, epochs=100):
    model.train()
    grad_history = {'norms': [], 'lr': [], 'loss': [], 'ce': [], 'margin': []}
    base_lr = 1e-3
    lr = base_lr

    for epoch in range(epochs):
        epoch_loss = epoch_ce = epoch_margin = 0.0
        epoch_grad_norms = []

        progress = epoch / epochs
        if progress > 0.7:
            ce_weight, margin_weight = 0.5, 0.5
        elif progress > 0.4:
            ce_weight, margin_weight = 0.6, 0.4
        else:
            ce_weight, margin_weight = 0.8, 0.2

        for batch_idx, (drug_pairs, labels) in enumerate(train_loader):
            drug_pairs = drug_pairs.to(device)
            labels = labels.to(device)
            drug1 = drug_pairs[:, 0, :]
            drug2 = drug_pairs[:, 1, :]

            optimizer.zero_grad()

            probs, digit_caps, logits = model._forward(drug1, drug2)
            ce_loss = criterion(logits, labels)
            num_classes = digit_caps.size(1)
            margin_loss = margin_criterion(
                digit_caps, F.one_hot(labels, num_classes=num_classes).float()
            )
            total_loss = ce_weight * ce_loss + margin_weight * margin_loss

            total_loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm **= 0.5
            epoch_grad_norms.append(total_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_ce += ce_loss.item()
            epoch_margin += margin_loss.item()

        avg_grad_norm = np.mean(epoch_grad_norms)
        grad_history['norms'].append(avg_grad_norm)
        grad_history['lr'].append(lr)
        grad_history['loss'].append(epoch_loss / len(train_loader))
        grad_history['ce'].append(epoch_ce / len(train_loader))
        grad_history['margin'].append(epoch_margin / len(train_loader))

        print(f"Epoch {epoch+1:03d} | Loss: {grad_history['loss'][-1]:.4f} | "
              f"CE: {grad_history['ce'][-1]:.4f} | Margin: {grad_history['margin'][-1]:.4f} | "
              f"LR: {lr:.6f} | GradNorm: {avg_grad_norm:.4f}")

    plot_gradient_history(grad_history)
    return grad_history


def plot_gradient_history(history):

    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['norms'], label='Gradient Norm')
    plt.title('Gradient Norm')
    plt.xlabel('Epoch')
    plt.ylabel('Norm')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(history['lr'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(history['loss'], label='Total Loss')
    plt.plot(history['ce'], label='CE Loss')
    plt.plot(history['margin'], label='Margin Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(np.log10(history['lr']), np.log10(history['norms']), 'o-')
    plt.title('LR vs Gradient Norm (log-log)')
    plt.xlabel('log10(LR)')
    plt.ylabel('log10(GradNorm)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_dynamics.png')
    plt.close()

# ===================== Evaluation Metrics =====================
def evaluate_model(model, test_loader, device, num_classes=3):
    model.eval()
    all_logits, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for drug_pairs, labels in test_loader:
            drug_pairs = drug_pairs.to(device)
            labels = labels.to(device)
            drug1 = drug_pairs[:, 0, :]
            drug2 = drug_pairs[:, 1, :]

            logits = model(drug1, drug2, return_logits=True)
            probs = torch.softmax(logits, dim=1)

            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    pred_labels = torch.argmax(all_logits, dim=1).numpy()
    true_labels = all_labels.numpy()

    metrics = {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'f1_weighted': f1_score(true_labels, pred_labels, average='weighted'),
        'f1_macro': f1_score(true_labels, pred_labels, average='macro'),
        'precision_weighted': precision_score(true_labels, pred_labels, average='weighted', zero_division=0),
        'recall_weighted': recall_score(true_labels, pred_labels, average='weighted'),
        'auprc_weighted': average_precision_score(true_labels, all_probs.numpy(), average='weighted'),
    }

    #需要把这部分ROC改动
    try:
        roc_auc_scores = []
        for i in range(num_classes):
            class_true = (true_labels == i).astype(int)
            class_probs = all_probs[:, i].numpy()
            if len(np.unique(class_true)) > 1:
                roc_auc_scores.append(roc_auc_score(class_true, class_probs))
        metrics['roc_auc_avg'] = np.mean(roc_auc_scores) if len(roc_auc_scores) > 0 else float('nan')
    except Exception as e:
        print(f"ROC AUC: {e}")
        metrics['roc_auc_avg'] = float('nan')

    return metrics, all_probs.numpy(), pred_labels


# ===================== 5-Fold Cross Validation =====================
def run_5fold_cv(X, Y, feature_type, device, epochs=100):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    
    feature_config = {
        '1D': {'dim': 128, 'start': 0},
        '2D': {'dim': 128, 'start': 128},
        '3D': {'dim': 128, 'start': 256},
        'bert': {'dim': 300, 'start': 384}
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold+1}/5 ===")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        Y_train_fold, Y_val_fold = Y[train_idx], Y[val_idx]
        
        train_loader = DataLoader(TensorDataset(X_train_fold, Y_train_fold), 
                                 batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_fold, Y_val_fold), 
                               batch_size=32, shuffle=False)
        
        model = DrugInteractionModel(
            feature_config=feature_config,
            hidden_dim=64,
            num_classes=3,
            temperature=0.5
        ).to(device)
        
        criterion = nn.CrossEntropyLoss().to(device)
        margin_criterion = model.margin_loss
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        print(f"Training on {len(train_idx)} samples...")
        train_model(model, train_loader, optimizer, criterion, margin_criterion, device, epochs=epochs)
        
        print(f"Evaluating on {len(val_idx)} samples...")
        metrics, all_probs, pred_labels = evaluate_model(model, val_loader, device)
        
        print(f"Fold {fold+1} Results:")
        print(f"Accuracy={metrics['accuracy']:.4f}")
        print(f"F1_weighted={metrics['f1_weighted']:.4f}")
        print(f"F1_macro={metrics['f1_macro']:.4f}")
        print(f"Precision_weighted={metrics['precision_weighted']:.4f}")
        print(f"Recall_weighted={metrics['recall_weighted']:.4f}")
        print(f"AUC-ROC={metrics['roc_auc_avg']:.4f}")
        print(f"AUPRC={metrics['auprc_weighted']:.4f}")
        
        fold_results.append(metrics)
    
    avg_metrics = {}
    for metric in fold_results[0].keys():
        values = [r[metric] for r in fold_results]
        avg_metrics[metric] = np.mean(values)
    
    print("\n=== 5-Fold Cross Validation Summary ===")
    for metric, value in avg_metrics.items():
        print(f"Average {metric}: {value:.4f}")
    
    return avg_metrics

# ===================== Main =====================
if __name__ == "__main__":
    print(f"当前使用的设备: {'GPU' if torch.cuda.is_available() else 'CPU'} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train = pd.read_csv('data/drug_dataset/ThirdStage_train.csv')
    for i in range(len(train)):
        if train.iloc[i,4] == 'Major':
            train.iloc[i,4] = 0
        if train.iloc[i,4] == 'Moderate':
            train.iloc[i,4] = 1
        if train.iloc[i,4] == 'Minor':
            train.iloc[i,4] = 2

    test = pd.read_csv('data/drug_dataset/ThirdStage_test.csv')
    for i in range(len(test)):
        if test.iloc[i,4] == 'Major':
            test.iloc[i,4] = 0
        if test.iloc[i,4] == 'Moderate':
            test.iloc[i,4] = 1
        if test.iloc[i,4] == 'Minor':
            test.iloc[i,4] = 2
    
    all_data = pd.concat([train, test], ignore_index=True)
    
    alldrug, features = process_dataset(train_data=all_data, feature_type=args.feature, device=device)
    
    X, Y = [], []
    for i in range(len(all_data)):
        drug1_id = all_data.iloc[i, 0]
        drug2_id = all_data.iloc[i, 2]
        X.append([alldrug[drug1_id], alldrug[drug2_id]])
        Y.append(all_data.iloc[i, 4])
    
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.long)
    
    # print("\n=== Starting 5-Fold Cross Validation ===")
    # cv_results = run_5fold_cv(X, Y, args.feature, device, epochs=100)
    
    print("\n=== Training Final Model on Entire Dataset ===")
    train_loader = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=True)
    
    feature_config = {
        '1D': {'dim': 128, 'start': 0},
        '2D': {'dim': 128, 'start': 128},
        '3D': {'dim': 128, 'start': 256},
        'bert': {'dim': 300, 'start': 384}
    }

    model = DrugInteractionModel(
        feature_config=feature_config,
        hidden_dim=64,
        num_classes=3,
        temperature=0.5
    ).to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    margin_criterion = model.margin_loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("Training configuration:")
    print(f"Optimizer: AdamW(lr=1e-3, weight_decay=1e-4)")
    print(f"Learning rate: Fixed at 1e-3")
    print(f"Loss weights: Dynamic adjustment based on training progress")
    
    print("Training final model...")
    train_model(model, train_loader, optimizer, criterion, margin_criterion, device, epochs=100)
    
    model_save_path = 'model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel weights saved to {model_save_path}")
    print("\n=== Evaluating Final Model on Entire Dataset ===")
    
    test_loader = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=False)
    
    metrics, all_probs, pred_labels = evaluate_model(model, test_loader, device)
    
    print("\n=== Final Model Performance ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1_weighted: {metrics['f1_weighted']:.4f}")
    print(f"F1_macro: {metrics['f1_macro']:.4f}")
    print(f"Precision_weighted: {metrics['precision_weighted']:.4f}")
    print(f"Recall_weighted: {metrics['recall_weighted']:.4f}")
    print(f"AUC-ROC: {metrics['roc_auc_avg']:.4f}")
    print(f"AUPRC: {metrics['auprc_weighted']:.4f}")
    
    results_df = pd.DataFrame({
        'Drug1_ID': all_data.iloc[:, 0],
        'Drug2_ID': all_data.iloc[:, 2],
        'True_Severity': all_data.iloc[:, 4],
        'Predicted_Severity': pred_labels,
        'Prob_Class0': all_probs[:, 0],
        'Prob_Class1': all_probs[:, 1],
        'Prob_Class2': all_probs[:, 2]
    })
    
    results_df.to_csv('final_model_predictions.csv', index=False)
    print("\n prediction results saved to final_model_predictions.csv")
    
    print("\n=== Starting SHAP Analysis (100 drug pairs) ===")

    feature_config = {
        '1D': {'dim': 128, 'start': 0},
        '2D': {'dim': 128, 'start': 128},
        '3D': {'dim': 128, 'start': 256},
        'bert': {'dim': 300, 'start': 384}  
    }

    print("\n=== Data Validation ===")
    print(f"Test data shape: {X.shape}")
    print(f"Total configured dimensions: {sum(v['dim'] for v in feature_config.values())}")

    analyzer = DrugSHAPAnalyzer(
        model=model,
        feature_config=feature_config,
        device=device
    )

    print("\nCalculating SHAP values (safe mode)...")
    shap_values = analyzer.compute_shap(
        drug_pairs=X[:100].numpy(),  
        nsamples=100,  
        batch_size=2   
    )

    df_summary = analyzer.save_shap_summary_to_csv("detailed_shap_analysis.csv")

    if not np.isnan(shap_values).all():
        print("\nGenerating visualizations...")
        analyzer.plot_comprehensive_analysis()
        
        shap_drugA = analyzer.shap_values[:, 0, :]
        shap_drugB = analyzer.shap_values[:, 1, :]
        
        def analyze_view_shap(shap_data, drug_name):
            print(f"\n=== {drug_name} Feature View Analysis ===")
            for name, config in feature_config.items():
                start, end = config['start'], config['start'] + config['dim']
                view_shap = shap_data[:, start:end]
                
                view_contrib = np.nanmean(np.abs(view_shap))
                print(f"{name} view average contribution: {view_contrib:.4f}")
                
                mean_abs = np.nanmean(np.abs(view_shap), axis=0)
                top_indices = np.argsort(mean_abs)[::-1][:3]
                for idx in top_indices:
                    feat_name = f"{name}_{idx}"
                    mean_shap = np.nanmean(view_shap[:, idx])
                    direction = "positive" if mean_shap > 0 else "negative"
                    print(f"  {feat_name}: {mean_abs[idx]:.4f} ({direction} impact)")
        
        analyze_view_shap(shap_drugA, "Drug A")
        analyze_view_shap(shap_drugB, "Drug B")
        
        total_A = np.nanmean(np.abs(shap_drugA))
        total_B = np.nanmean(np.abs(shap_drugB))
        print(f"\nTotal Contribution | Drug A: {total_A:.4f} | Drug B: {total_B:.4f}")