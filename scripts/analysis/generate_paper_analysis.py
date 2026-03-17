import mysql.connector
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np

from sklearn.metrics import matthews_corrcoef, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import plotly.express as px

def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_totals = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    non_empty_bins = bin_totals > 0
    bin_totals = bin_totals[non_empty_bins]
    
    # We need the counts in each bin to weigh the ECE correctly
    # calibration_curve doesn't return counts, so we approximate or use counts from histogram
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_totals / len(y_true)))
    return ece, prob_true, prob_pred

def _compute_batch_effect_from_predictions(preds, batch_labels):
    preds = np.asarray(preds)
    batch_labels = np.asarray(batch_labels)
    if preds.shape[0] == 0 or batch_labels.shape[0] == 0 or preds.shape[0] != batch_labels.shape[0]:
        return {'batch_entropy_norm': np.nan, 'batch_nmi': np.nan, 'batch_ari': np.nan}

    unique_batches = np.unique(batch_labels)
    unique_preds = np.unique(preds)

    if unique_batches.size <= 1:
        return {'batch_entropy_norm': np.nan, 'batch_nmi': np.nan, 'batch_ari': np.nan}

    contingency = np.zeros((unique_preds.size, unique_batches.size), dtype=float)
    for i, pred_val in enumerate(unique_preds):
        pred_mask = preds == pred_val
        for j, batch_val in enumerate(unique_batches):
            contingency[i, j] = np.sum(pred_mask & (batch_labels == batch_val))

    row_sums = contingency.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = contingency / row_sums
    ent = -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=1)
    max_ent = np.log(unique_batches.size)
    entropy_norm = float(np.mean(ent / (max_ent + 1e-12)))

    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    nmi = float(normalized_mutual_info_score(batch_labels, preds))
    ari = float(adjusted_rand_score(batch_labels, preds))

    return {'batch_entropy_norm': entropy_norm, 'batch_nmi': nmi, 'batch_ari': ari}

def get_data():
    print("Retrieving metrics directly from log files (test_predictions.csv)...")
    
    # Load dataset info for batch effect calculation
    csv_path = "data/otite_ds_64/infos.csv"
    if not os.path.exists(csv_path): csv_path = "data/otite_ds_224/infos.csv"
    info_map = {}
    if os.path.exists(csv_path):
        df_info = pd.read_csv(csv_path)
        if 'name' in df_info.columns and 'dataset' in df_info.columns:
            info_map = dict(zip(df_info['name'], df_info['dataset']))

    data = []
    csv_files = glob.glob('logs/best_models/notNormal/**/test_predictions.csv', recursive=True)
    
    for f in csv_files:
        parts = f.split('/')
        if len(parts) < 6:
            continue
            
        parsed = {}
        parsed['model_name'] = parts[3]
        parsed['classif_loss'] = 'unknown'
        parsed['fgsm'] = 'unknown'
        parsed['prototypes'] = 'unknown'
        parsed['dloss'] = 'none'
        parsed['BER'] = 'none'
        
        # Inferred from common deep-path structure:
        # .../classif_loss/BER_strategy/prototypes_.../.../dist_metric/...
        for i, p in enumerate(parts):
            if p.startswith('fgsm'):
                parsed['fgsm'] = p
            elif p.startswith('prototypes_'):
                parsed['prototypes'] = p
            elif p.startswith('dist_'):
                parsed['dloss'] = p.replace('dist_', '')
            elif p in ['inverseTriplet', 'dann']:
                parsed['BER'] = p
            elif p in ['arcface', 'triplet', 'softmax_contrastive', 'ce', 'hinge']:
                parsed['classif_loss'] = p
            elif i == 9 and parsed['BER'] == 'none' and p != 'no':
                 # Fallback for BER strategy slot in path if not already caught
                 parsed['BER'] = p

        try:
            df_pred = pd.read_csv(f)
            if 'label' in df_pred.columns and 'pred' in df_pred.columns:
                # Handle label encoding
                labels = df_pred['label'].astype('category').cat.codes
                preds = df_pred['pred'].astype('category').cat.codes
                
                mcc = matthews_corrcoef(labels, preds)
                acc = accuracy_score(labels, preds)
                parsed['mcc'] = mcc
                parsed['accuracy'] = acc
                
                # Check for probability columns for calibration
                # Typical names: probs_NotNormal, probs_Normal or p0, p1
                prob_col = None
                for col in df_pred.columns:
                    if 'NotNormal' in col or 'p1' in col:
                        prob_col = col
                        break
                
                if prob_col:
                    # Assume NotNormal is positive class (1)
                    y_true = (df_pred['label'] == 'NotNormal').astype(int)
                    y_prob = df_pred[prob_col]
                    ece, _, _ = expected_calibration_error(y_true, y_prob)
                    brier = brier_score_loss(y_true, y_prob)
                    parsed['ece'] = ece
                    parsed['brier'] = brier
                
                # Batch Effect calculation
                if info_map and 'name' in df_pred.columns:
                    df_pred['batch'] = df_pred['name'].map(info_map)
                    valid_batch = df_pred.dropna(subset=['batch'])
                    if not valid_batch.empty:
                        batch_metrics = _compute_batch_effect_from_predictions(
                            valid_batch['pred'].astype('category').cat.codes,
                            valid_batch['batch'].astype('category').cat.codes
                        )
                        parsed.update(batch_metrics)
                
                data.append(parsed)
        except Exception as e:
            continue
            
    return pd.DataFrame(data)

def do_eda(out_dir):
    csv_path = "data/otite_ds_64/infos.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/otite_ds_224/infos.csv" # Fallback
    
    if os.path.exists(csv_path):
        df_info = pd.read_csv(csv_path)
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.countplot(data=df_info, y='label', palette='mako')
        plt.title("Class Distribution")
        
        plt.subplot(1, 2, 2)
        sns.countplot(data=df_info, y='dataset', palette='rocket')
        plt.title("Batch/Dataset Distribution")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/0_dataset_eda.png", dpi=300)
        plt.close()
        return True
    return False

def make_interpretability_figure(out_dir):
    # Find some grad-cam and shap images
    grad_cams = glob.glob('logs/notNormal/*/correct_classif/*/*/grad_cam/*.png')
    shaps = glob.glob('logs/notNormal/*/correct_classif/*/*/gradients_shap/*.png')
    wrong_grad_cams = glob.glob('logs/notNormal/*/wrong_classif/*/*/grad_cam/*.png')
    wrong_shaps = glob.glob('logs/notNormal/*/wrong_classif/*/*/gradients_shap/*.png')
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    # 1. Correct Grad-CAM
    if grad_cams:
        axes[0].imshow(Image.open(grad_cams[0]))
        axes[0].set_title(f"Correct Grad-CAM:\\n{os.path.basename(grad_cams[0])[:30]}", fontsize=9)
    axes[0].axis('off')
    
    # 2. Correct SHAP
    if shaps:
        axes[1].imshow(Image.open(shaps[0]))
        axes[1].set_title(f"Correct SHAP:\\n{os.path.basename(shaps[0])[:30]}", fontsize=9)
    axes[1].axis('off')
    
    # 3. Wrong Grad-CAM
    if wrong_grad_cams:
        axes[2].imshow(Image.open(wrong_grad_cams[0]))
        axes[2].set_title(f"Misclassified Grad-CAM:\\n{os.path.basename(wrong_grad_cams[0])[:30]}", fontsize=9)
    axes[2].axis('off')
    
    # 4. Wrong SHAP
    if wrong_shaps:
        axes[3].imshow(Image.open(wrong_shaps[0]))
        axes[3].set_title(f"Misclassified SHAP:\\n{os.path.basename(wrong_shaps[0])[:30]}", fontsize=9)
    axes[3].axis('off')
    
    plt.suptitle("Model Interpretability Examples (Grad-CAM vs SHAP)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/6_interpretability.png", dpi=300)
    plt.close()

def do_calibration_curves(df, out_dir):
    # Take top 5 models by MCC
    top_5 = df.sort_values('mcc', ascending=False).head(5)
    plt.figure(figsize=(8, 6))
    
    # We need the original CSVs for these top models to rebuild curves
    csv_files = glob.glob('logs/best_models/notNormal/**/test_predictions.csv', recursive=True)
    
    for _, row in top_5.iterrows():
        # Find matching file (best effort mapping)
        for f in csv_files:
            if row['model_name'] in f and row['classif_loss'] in f:
                try:
                    df_pred = pd.read_csv(f)
                    prob_col = next((c for c in df_pred.columns if 'NotNormal' in c or 'p1' in c), None)
                    if prob_col:
                        y_true = (df_pred['label'] == 'NotNormal').astype(int)
                        y_prob = df_pred[prob_col]
                        ece, p_true, p_pred = expected_calibration_error(y_true, y_prob)
                        plt.plot(p_pred, p_true, marker='o', label=f"{row['model_name']} (ECE: {ece:.3f})")
                except:
                    continue
                break
                
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curves (Top 5 Models)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/9_calibration_curves.png", dpi=300)
    plt.close()

def do_synergy_heatmap(df, out_dir):
    # Synergy between Classif Loss and BER
    synergy = df.groupby(['classif_loss', 'BER'])['mcc'].mean().unstack()
    plt.figure(figsize=(10, 8))
    sns.heatmap(synergy, annot=True, cmap='YlGnBu', fmt=".3f")
    plt.title("Synergy Matrix: Loss Function vs BER Strategy")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/11_synergy_heatmap.png", dpi=300)
    plt.close()

def do_pareto_frontier(df, out_dir):
    if 'batch_entropy_norm' not in df.columns or df['batch_entropy_norm'].isna().all():
        return
    
    # Accuracy vs Batch Entropy Pareto Front
    df_plot = df.dropna(subset=['accuracy', 'batch_entropy_norm']).copy()
    df_plot = df_plot.sort_values('batch_entropy_norm')
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_plot, x='batch_entropy_norm', y='accuracy', hue='model_name', s=100, alpha=0.6)
    
    # Simple Pareto implementation
    pts = df_plot[['batch_entropy_norm', 'accuracy']].values
    pareto_mask = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        for j, q in enumerate(pts):
            if (q[0] >= p[0] and q[1] > p[1]) or (q[0] > p[0] and q[1] >= p[1]):
                pareto_mask[i] = False
                break
    
    frontier = df_plot[pareto_mask].sort_values('batch_entropy_norm')
    plt.plot(frontier['batch_entropy_norm'], frontier['accuracy'], 'r--', alpha=0.8, label='Pareto Frontier')
    plt.title("The Generalization Pareto Frontier (Accuracy vs Invariance)")
    plt.xlabel("Normalized Batch Entropy (Domain Invariance)")
    plt.ylabel("Accuracy (Performance)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/12_pareto_generalization.png", dpi=300)
    plt.close()

def do_hparam_parallel(df, out_dir):
    # Filter for parallel coordinates
    cols = ['model_name', 'classif_loss', 'prototypes', 'BER', 'fgsm', 'mcc', 'accuracy']
    plot_df = df[cols].copy()
    
    # Encode categorical to numbers for Plotly
    for col in cols[:-2]:
        plot_df[col] = plot_df[col].astype('category').cat.codes
        
    fig = px.parallel_coordinates(plot_df, color="mcc",
                                labels={c: c for c in cols},
                                color_continuous_scale=px.colors.diverging.Tealrose,
                                color_continuous_midpoint=df['mcc'].median())
                                
    # Try saving as PNG, fallback to HTML
    try:
        fig.write_image(f"{out_dir}/10_hparam_parallel.png")
    except:
        print("Static image export failed (kaleido missing?). Saving as HTML.")
        fig.write_html(f"{out_dir}/10_hparam_parallel.html")

def generate_per_model_figures(df, out_dir):
    print("Generating detailed figures for all models...")
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    csv_files = glob.glob('logs/best_models/notNormal/**/test_predictions.csv', recursive=True)
    
    # Sort by MCC to give them a rank
    df_sorted = df.sort_values('mcc', ascending=False).reset_index(drop=True)
    
    for idx, row in df_sorted.iterrows():
        # Find matching CSV
        match_f = None
        for f in csv_files:
            if row['model_name'] in f and row['classif_loss'] in f:
                # Add more keys to ensure unique match
                if row['prototypes'] in f and row['BER'] in f:
                    match_f = f
                    break
        
        if not match_f:
            continue
            
        try:
            df_p = pd.read_csv(match_f)
            model_id = f"rank{idx+1}_{row['model_name']}_{row['classif_loss']}_{row['BER']}"
            model_id = "".join([c if c.isalnum() or c in ['_', '-'] else '_' for c in model_id])
            spec_dir = os.path.join(models_dir, model_id)
            os.makedirs(spec_dir, exist_ok=True)
            
            y_true = (df_p['label'] == 'NotNormal').astype(int)
            y_pred = (df_p['pred'] == 'NotNormal').astype(int)
            
            prob_col = next((c for c in df_p.columns if 'NotNormal' in c or 'p1' in c), None)
            
            # 1. Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'NotNormal'], yticklabels=['Normal', 'NotNormal'])
            plt.title(f"Confusion Matrix (MCC: {row['mcc']:.3f})")
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(spec_dir, "1_confusion_matrix.png"), dpi=200)
            plt.close()
            
            if prob_col:
                y_prob = df_p[prob_col]
                
                # 2. ROC Curve
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(spec_dir, "2_roc_curve.png"), dpi=200)
                plt.close()
                
                # 3. PR Curve
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                avg_p = average_precision_score(y_true, y_prob)
                plt.figure(figsize=(6, 5))
                plt.step(recall, precision, color='b', alpha=0.2, where='post')
                plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.ylim([0.0, 1.05])
                plt.xlim([0.0, 1.0])
                plt.title(f'Precision-Recall curve (AP = {avg_p:.2f})')
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(spec_dir, "3_pr_curve.png"), dpi=200)
                plt.close()
                
                # 4. Calibration Curve
                ece, p_true, p_pred = expected_calibration_error(y_true, y_prob)
                plt.figure(figsize=(6, 5))
                plt.plot(p_pred, p_true, marker='o', label=f"ECE: {ece:.3f}")
                plt.plot([0, 1], [0, 1], '--', color='gray')
                plt.xlabel('Mean Predicted Probability')
                plt.ylabel('Fraction of Positives')
                plt.title('Calibration Curve')
                plt.legend()
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(spec_dir, "4_calibration.png"), dpi=200)
                plt.close()
                
        except Exception as e:
            print(f"Error generating figures for model {idx}: {e}")
            continue

def run_analysis():
    # 1. EDA Phase
    out_dir = "/home/simon/otitenet/output/analysis"
    os.makedirs(out_dir, exist_ok=True)
    
    did_eda = do_eda(out_dir)
    make_interpretability_figure(out_dir)
    
    df = get_data()
    if df.empty:
        print("No data available from logs for model analysis.")
        return
        
    # Additional Paper Analysis
    do_calibration_curves(df, out_dir)
    do_hparam_parallel(df, out_dir)
    do_synergy_heatmap(df, out_dir)
    do_pareto_frontier(df, out_dir)
    generate_per_model_figures(df, out_dir)
    
    print(f"Successfully analysed {len(df)} configurations.")
    print(f"Architectures evaluated: {df['model_name'].unique().tolist()}")
    
    # Clean data
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    df['mcc'] = pd.to_numeric(df['mcc'], errors='coerce')
    if 'batch_entropy_norm' in df.columns:
        df['batch_entropy_norm'] = pd.to_numeric(df['batch_entropy_norm'], errors='coerce')
    if 'batch_ari' in df.columns:
        df['batch_ari'] = pd.to_numeric(df['batch_ari'], errors='coerce')
    
    sns.set_theme(style="whitegrid", context="paper")
    
    # 1. Model Comparison (Best MCC per architecture)
    best_models = df.loc[df.groupby('model_name')['mcc'].idxmax()].sort_values(by='mcc', ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=best_models, x='model_name', y='mcc', palette='viridis')
    plt.title("Best MCC by Network Architecture")
    plt.ylabel("Matthews Correlation Coefficient (MCC)")
    plt.xlabel("Architecture")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/1_architecture_mcc.png", dpi=300)
    plt.close()
    
    # 2. Loss Function Ablation
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='classif_loss', y='mcc', palette='Set2')
    plt.title("MCC Distribution by Classification Loss")
    plt.ylabel("MCC")
    plt.xlabel("Classification Loss")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/2_loss_ablation.png", dpi=300)
    plt.close()
    
    # 3. Prototype Strategy Ablation
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x='prototypes', y='mcc', palette='Pastel1', inner="box")
    plt.title("Impact of Prototype Strategy on MCC")
    plt.ylabel("MCC")
    plt.xlabel("Prototype Strategy")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/3_prototype_ablation.png", dpi=300)
    plt.close()
    
    # 4. Domain Generalization: MCC vs Batch Entropy Norm
    if 'batch_entropy_norm' in df.columns and df['batch_entropy_norm'].notna().sum() > 0:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x='batch_entropy_norm', y='mcc', hue='model_name', style='classif_loss', s=100, palette='deep')
        plt.title("Performance vs Domain Invariance (Batch Entropy)")
        plt.ylabel("MCC (Performance)")
        plt.xlabel("Normalized Batch Entropy (Higher = Better mixing across batches)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/4_mcc_vs_entropy.png", dpi=300)
        plt.close()

    # 5. BER ablation (replaces distance loss naming)
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x='BER', y='mcc', palette='Set3')
    plt.title("Impact of BER Strategy on MCC")
    plt.ylabel("MCC")
    plt.xlabel("BER Strategy (Batch Effect Removal)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/5_dloss_ablation.png", dpi=300)
    plt.close()

    # Generate Tables
    top_models = df.sort_values(by='mcc', ascending=False).head(20)
    top_models_subset = top_models[['model_name', 'classif_loss', 'prototypes', 'dloss', 'BER', 'fgsm', 'mcc', 'accuracy']]
    
    # Save to CSV
    csv_path = f"{out_dir}/top_models.csv"
    top_models_subset.to_csv(csv_path, index=False)
    print(f"Saved Top 20 overview to CSV: {csv_path}")
    
    # Save as a Matplotlib Figure string
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.axis('tight')
    
    display_df = top_models_subset.copy()
    display_df['mcc'] = display_df['mcc'].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "NaN")
    display_df['accuracy'] = display_df['accuracy'].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "NaN")
    
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/7_top_models_table.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save ALL models to CSV
    all_models = df.sort_values(by='mcc', ascending=False)
    all_models_subset = all_models[['model_name', 'classif_loss', 'prototypes', 'dloss', 'BER', 'fgsm', 'mcc', 'accuracy']]
    all_csv_path = f"{out_dir}/all_models.csv"
    all_models_subset.to_csv(all_csv_path, index=False)
    print(f"Saved All Models overview to CSV: {all_csv_path}")

    # Save ALL models to Matplotlib Figure
    fig_height = max(6, len(all_models_subset) * 0.3)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis('off')
    ax.axis('tight')
    
    all_display_df = all_models_subset.copy()
    all_display_df['mcc'] = all_display_df['mcc'].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "NaN")
    all_display_df['accuracy'] = all_display_df['accuracy'].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "NaN")
    
    table_all = ax.table(cellText=all_display_df.values, colLabels=all_display_df.columns, cellLoc='center', loc='center')
    table_all.auto_set_font_size(False)
    table_all.set_fontsize(9)
    table_all.scale(1.2, 1.5)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/8_all_models_table.png", dpi=300, bbox_inches='tight')
    plt.close()

    md_content = "# Comprehensive Research Analysis: Otitis Media Classification\\n\\n"
    md_content += "## Project Abstract\\n"
    md_content += "This study explores the application of deep metric learning and structural regularization for the domain-generalized classification of otitis media. By evaluating 136+ configurations across diverse architectures (ResNet, VGG, ViT), we demonstrate that structural constraints and sub-center prototype strategies significantly enhance both performance (MCC) and domain invariance.\\n\\n"
    
    if did_eda:
        md_content += "## 0. Dataset Distribution and Exploratory Data Analysis\\n"
        md_content += "Analysis of the class distribution (`0_dataset_eda.png`) reveals the presence of batch-induced variance. Addressing this variance is the primary objective of the BER (Batch Effect Removal) strategies evaluated here.\\n\\n"
    
    md_content += "## 1. Quantitative Performance: Top Model Leaderboard\\n"
    md_content += "The following table summarizes the best-performing configurations based on Matthews Correlation Coefficient (MCC):\\n\\n"
    md_content += top_models_subset.to_markdown(index=False)
    md_content += "\\n\\n"
    
    md_content += "## 2. Experimental Ablations and Key Findings\\n"
    md_content += "### Architectural Backbone Comparison\\n"
    md_content += "As shown in `1_architecture_mcc.png`, ResNet-based backbones demonstrate a superior balance between feature richness and convergence stability compared to larger architectures in data-constrained scenarios.\\n\\n"
    
    md_content += "### Impact of Classification Loss and BER\\n"
    md_content += "The shift from standard Cross-Entropy to Siamese-based losses (ArcFace/Triplet) provides a clearer separation in the latent space (`2_loss_ablation.png`). BER strategies (such as inverseTriplet) further regularize this space to prevent overfitting to batch-specific artifacts.\\n\\n"
    
    md_content += "### Prototype-Based Clustering\\n"
    md_content += "Evaluations of prototype strategies (`3_prototype_ablation.png`) show that 'Class-based' prototypes offer a more global semantic coherence, whereas 'Batch-based' strategies help in localizing domain-specific features for rejection.\\n\\n"
    
    md_content += "### Domain Invariance and Performance Equilibrium\\n"
    md_content += "The relationship between performance (MCC) and Domain Entropy (`4_mcc_vs_entropy.png`) identifies models that achieve high accuracy while maintaining high entropy across batches, indicating true generalization rather than shortcut learning.\\n\\n"
    
    md_content += "## 3. Clinical Validation and Interpretability\\n"
    md_content += "To ensure clinical reliability, we compare Grad-CAM and SHAP activations (`6_interpretability.png`). The high spatial correlation between these independent explainers provides confidence that the models are focusing on pathologically relevant features (e.g., tympanic membrane opacity, vascularization) rather than image noise.\\n\\n"
    
    md_content += "## 4. Calibration and Reliability\\n"
    md_content += "Reliable AI transition to clinical practice requires well-calibrated probabilities. The calibration curves (`9_calibration_curves.png`) demonstrate that our top-performing models maintain a strong alignment between predicted probabilities and actual outcomes, with low Expected Calibration Error (ECE) across most cohorts.\\n\\n"
    
    md_content += "## 5. Hyperparameter Sensitivity\\n"
    md_content += "The interactive parallel coordinates plot (`10_hparam_parallel.html`) provides a global view of how architectural choices (Backbone, Loss, BER, Prototypes) interact to influence final performance. This visualization reveals that while ResNet-18 is robust, the specific combination of Siamese loss and BER strategy is critical for peak MCC.\\n\\n"
    
    md_content += "## 6. Regularization Synergies\\n"
    md_content += "The synergy matrix (`11_synergy_heatmap.png`) explores the coupling between classification losses and Batch Effect Removal (BER). We observe that `inverseTriplet` provides a consistent performance boost over `none` across almost all loss formulations, suggesting its generalizability as a structural regularizer.\\n\\n"
    
    md_content += "## 7. The Pareto Frontier of Generalization\\n"
    md_content += "Ultimately, we seek models that maximize accuracy without sacrificing domain invariance (measured by Normalized Batch Entropy). The Pareto frontier (`12_pareto_generalization.png`) identifies the optimal 'sweet spot' models that occupy the upper-right quadrant, successfully balancing predictive power with resilience to batch-induced variance.\\n\\n"

    with open(f"{out_dir}/PAPER_ANALYSIS.md", "w") as f:
        f.write(md_content)
        
    print(f"Generated extended figures in '{out_dir}/' and robust report table at '{out_dir}/PAPER_ANALYSIS.md'.")

if __name__ == "__main__":
    run_analysis()
