import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Path to your completed runs CSV
COMPLETED_CSV = 'logs/progresses/notNormal/PROD_notNormal_completed_runs_metrics.csv'
OUT_DIR = 'output/analysis'

os.makedirs(OUT_DIR, exist_ok=True)

# Read the data
df = pd.read_csv(COMPLETED_CSV)

# Filter out failed/incomplete runs if needed
df = df[df['status'].str.lower().str.startswith('completed')]

# --- 1. Barplot with error bars (std) by architecture ---
if 'model_name' in df.columns:
    arch_col = 'model_name'
else:
    arch_col = 'variant'  # fallback

arch_stats = df.groupby(arch_col)['valid_mcc'].agg(['mean', 'std', 'count']).reset_index()

plt.figure(figsize=(10, 6))
# Use plt.bar (not sns.barplot) because seaborn barplot does not support yerr=
plt.bar(
    arch_stats[arch_col],
    arch_stats['mean'],
    yerr=arch_stats['std'].fillna(0),
    capsize=5,
    color=sns.color_palette('deep', n_colors=len(arch_stats)),
)
plt.ylabel('Matthews Correlation Coefficient (MCC)')
plt.xlabel('Architecture')
plt.title('Best MCC by Network Architecture (with Std Error Bars)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'mcc_by_architecture.png'), dpi=200)
plt.close()
print(f"Saved mcc_by_architecture.png to {OUT_DIR}")

# --- 2. Barplot for every unique model (full identifier) ---
if 'uuid' in df.columns:
    model_id_col = 'uuid'
elif 'run_dir' in df.columns:
    model_id_col = 'run_dir'
else:
    model_id_col = arch_col  # fallback

# (a) With error bars if multiple trials per model
model_stats = df.groupby(model_id_col)['valid_mcc'].agg(['mean', 'std', 'count']).reset_index()

plt.figure(figsize=(max(12, len(model_stats) * 0.5), 6))
plt.bar(
    range(len(model_stats)),
    model_stats['mean'],
    yerr=model_stats['std'].fillna(0),
    capsize=3,
)
plt.xticks(range(len(model_stats)), model_stats[model_id_col], rotation=90, fontsize=7)
plt.ylabel('Matthews Correlation Coefficient (MCC)')
plt.xlabel('Model (unique id)')
plt.title('MCC for Each Model (with Std Error Bars)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'mcc_by_model_with_errorbars.png'), dpi=200)
plt.close()
print(f"Saved mcc_by_model_with_errorbars.png to {OUT_DIR}")

# (b) Only top model per group (no error bar)
top_models = df.loc[df.groupby(model_id_col)['valid_mcc'].idxmax()]

plt.figure(figsize=(max(12, len(top_models) * 0.5), 6))
sns.barplot(data=top_models, x=model_id_col, y='valid_mcc')
plt.ylabel('Matthews Correlation Coefficient (MCC)')
plt.xlabel('Model (unique id)')
plt.title('Top MCC for Each Model (No Error Bar)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'top_mcc_by_model.png'), dpi=200)
plt.close()
print(f"Saved top_mcc_by_model.png to {OUT_DIR}")

print('Plots generated. You can adjust the model_id_col or arch_col as needed for your data.')
