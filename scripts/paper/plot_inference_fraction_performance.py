#!/usr/bin/env python3
"""Make performance-vs-inference-training-fraction figures for an experiment run."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-root", type=Path, required=True)
    p.add_argument("--metrics", type=Path, default=ROOT / "completed_runs_metrics.csv")
    p.add_argument("--metric", choices=("test_mcc", "valid_mcc", "test_acc", "valid_acc"), default="test_mcc")
    args=p.parse_args(); root=args.experiment_root
    catalog=pd.read_csv(root / "run_catalog.csv")
    completed=catalog[catalog.status.eq("completed") & catalog.uuid.notna() & catalog.uuid.astype(str).ne("")].copy()
    metrics=pd.read_csv(args.metrics)
    metrics["uuid"]=metrics["uuid"].astype(str)
    data=completed.merge(metrics, on="uuid", how="left", suffixes=("_planned", ""))
    data[args.metric]=pd.to_numeric(data[args.metric], errors="coerce")
    data["scenario_fraction"]=pd.to_numeric(data["scenario_fraction"], errors="coerce")
    data.to_csv(root / "performance_by_run.csv", index=False)
    fig_dir=root / "figures"; fig_dir.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax=plt.subplots(figsize=(8.5,5.4))
    for rank, group in data.dropna(subset=[args.metric]).groupby("rank"):
        group=group.sort_values("scenario_fraction")
        label=f"Best config rank {rank}"
        ax.plot(group.scenario_fraction, group[args.metric], marker="o", linewidth=2, label=label)
    mean=data.groupby("scenario_fraction", as_index=False)[args.metric].mean().sort_values("scenario_fraction")
    if not mean.empty: ax.plot(mean.scenario_fraction, mean[args.metric], color="black", linestyle="--", linewidth=2.5, marker="s", label="Mean selected best configs")
    ax.set(xlabel="Inference samples used for training (fraction of inference folder)", ylabel=args.metric.replace("_", " ").upper(), title="Performance gain as inference training data increases")
    ax.set_xticks(sorted(data.scenario_fraction.dropna().unique())); ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    fig.savefig(fig_dir / f"{args.metric}_by_train_fraction.png", dpi=220); fig.savefig(fig_dir / f"{args.metric}_by_train_fraction.pdf"); plt.close(fig)
    if not data.empty:
        pivot=data.pivot_table(index="scenario_fraction", columns="rank", values=args.metric, aggfunc="mean").sort_index()
        pivot.to_csv(root / f"{args.metric}_pivot.csv")
    print(f"Wrote {fig_dir / f'{args.metric}_by_train_fraction.png'}")
    return 0
if __name__ == "__main__": raise SystemExit(main())
