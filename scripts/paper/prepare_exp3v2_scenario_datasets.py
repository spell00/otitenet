#!/usr/bin/env python3
"""Prepare E03v2 Pareto scenario datasets.

Each output dataset keeps the four non-inference datasets in train. For the
inference/new-sample dataset, it fixes a proportional stratified 25% valid split
and 25% test split, then adds a scenario-specific proportional stratified train
subset of the inference samples (0%, 5%, 10%, 25%, or 50% by default). Remaining
inference rows are marked unused and ignored by GetData.split_from_infos_groups().
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "data/otite_ds_224/USA_Turquie_Chili_GMFUNL_inference_20260803"
DEFAULT_OUT_BASE = ROOT / "data/otite_ds_224"
NON_INFERENCE_DATASETS = [
    "Banque_Calaman_USA_2020_trie_CM",
    "Banque_Viscaino_Chili_2020",
    "Banque_Comert_Turquie_2020_jpg",
    "GMFUNL_jan2023",
]


def _scenario_label(frac: float) -> str:
    text = f"{frac:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p") or "0"


def _largest_remainder_counts(labels: list[str], counts: dict[str, int], n: int) -> dict[str, int]:
    total = sum(counts[label] for label in labels)
    if n <= 0:
        return {label: 0 for label in labels}
    if n > total:
        raise ValueError(f"Requested n={n} exceeds available samples ({total}).")
    quotas = {label: n * counts[label] / total for label in labels}
    out = {label: int(np.floor(quotas[label])) for label in labels}
    remaining = n - sum(out.values())
    for label in sorted(labels, key=lambda x: (quotas[x] - out[x], counts[x], x), reverse=True)[:remaining]:
        out[label] += 1
    # Keep every present class represented whenever n allows it.
    if n >= len(labels):
        for label in labels:
            if out[label] > 0:
                continue
            donor = max((x for x in labels if out[x] > 1), key=lambda x: (out[x] - quotas[x], out[x], x))
            out[label] = 1
            out[donor] -= 1
    return out


def _stratified_take(df: pd.DataFrame, n: int, seed: int) -> pd.Index:
    if n <= 0:
        return pd.Index([])
    labels = sorted(df["label"].astype(str).unique().tolist())
    counts = df["label"].astype(str).value_counts().to_dict()
    requested = _largest_remainder_counts(labels, {k: int(v) for k, v in counts.items()}, n)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for label in labels:
        pool = df[df["label"].astype(str).eq(label)]
        take = requested[label]
        if take == 0:
            continue
        if len(pool) < take:
            raise ValueError(f"Not enough {label} rows: need {take}, found {len(pool)}")
        rel = rng.choice(len(pool), size=take, replace=False)
        chosen.extend(pool.iloc[rel].index.tolist())
    return pd.Index(chosen)


def _link_or_copy_images(src_dir: Path, dst_dir: Path, names: pd.Series, mode: str) -> None:
    for name in sorted(set(names.astype(str).tolist())):
        src = src_dir / name
        dst = dst_dir / name
        if dst.exists() or dst.is_symlink():
            continue
        if not src.exists():
            raise FileNotFoundError(f"Missing image referenced by infos.csv: {src}")
        if mode == "symlink":
            os.symlink(os.path.relpath(src, dst_dir), dst)
        elif mode == "hardlink":
            os.link(src, dst)
        elif mode == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"Unknown image mode: {mode}")


def build_scenarios(base_dir: Path, out_base: Path, fractions: list[float], seed: int, prefix: str, image_mode: str, other_datasets: str) -> pd.DataFrame:
    infos_path = base_dir / "infos.csv"
    if not infos_path.exists():
        raise FileNotFoundError(infos_path)
    base = pd.read_csv(infos_path)
    required_cols = {"dataset", "name", "raw_label", "label", "group"}
    missing = required_cols - set(base.columns)
    if missing:
        raise ValueError(f"{infos_path} missing columns: {sorted(missing)}")

    inf = base[base["dataset"].eq("inference")].copy()
    if inf.empty:
        raise ValueError("No inference rows found in base infos.csv")
    total = len(inf)
    eval_n = int(np.floor(total * 0.25))

    test_idx = _stratified_take(inf, eval_n, seed + 1000)
    remaining_after_test = inf.drop(index=test_idx)
    valid_idx = _stratified_take(remaining_after_test, eval_n, seed + 2000)
    train_pool = inf.drop(index=test_idx.union(valid_idx))

    rows = []
    for frac in fractions:
        label = _scenario_label(frac)
        scenario_train_n = int(np.floor(total * frac))
        if scenario_train_n > len(train_pool):
            raise ValueError(
                f"Scenario {frac} requests {scenario_train_n} inference train rows, "
                f"but only {len(train_pool)} remain after fixed valid/test."
            )
        train_idx = _stratified_take(train_pool, scenario_train_n, seed + int(round(frac * 10000)) + 3000)

        out = base.copy()
        # Non-inference datasets can be kept in train (historical-data arm)
        # or marked unused (new-data-only arm). Inference rows are assigned below.
        other_group = "train" if other_datasets == "train" else "unused"
        out.loc[out["dataset"].isin(NON_INFERENCE_DATASETS), "group"] = other_group
        out.loc[out["dataset"].eq("inference"), "group"] = "unused"
        out.loc[test_idx, "group"] = "test"
        out.loc[valid_idx, "group"] = "valid"
        out.loc[train_idx, "group"] = "train"

        out_dir = out_base / f"{prefix}_train{label}_seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_dir / "infos.csv", index=False)
        _link_or_copy_images(base_dir, out_dir, out["name"], image_mode)

        counts = out.groupby(["dataset", "group", "label"]).size().reset_index(name="n")
        summary = {
            "scenario_fraction": frac,
            "scenario_label": label,
            "seed": seed,
            "dataset_path": str(out_dir.relative_to(ROOT)),
            "total_inference": total,
            "inference_train": int((out["dataset"].eq("inference") & out["group"].eq("train")).sum()),
            "inference_valid": int((out["dataset"].eq("inference") & out["group"].eq("valid")).sum()),
            "inference_test": int((out["dataset"].eq("inference") & out["group"].eq("test")).sum()),
            "inference_unused": int((out["dataset"].eq("inference") & out["group"].eq("unused")).sum()),
            "other_datasets": other_datasets,
            "total_train": int((out["group"].eq("train")).sum()),
            "counts": counts.to_dict(orient="records"),
        }
        (out_dir / "scenario_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        rows.append({k: v for k, v in summary.items() if k != "counts"})

    summary_df = pd.DataFrame(rows)
    manifest = out_base / f"{prefix}_seed{seed}_scenarios.csv"
    summary_df.to_csv(manifest, index=False)
    print(summary_df.to_string(index=False))
    print(f"Wrote scenario manifest: {manifest.relative_to(ROOT)}")
    return summary_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare E03v2 scenario datasets from a processed OtiteNet dataset.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE)
    parser.add_argument("--fractions", default="0,0.05,0.1,0.25,0.5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="USA_Turquie_Chili_GMFUNL_inference_exp3v2")
    parser.add_argument("--image-mode", choices=["symlink", "hardlink", "copy"], default="hardlink")
    parser.add_argument("--other-datasets", choices=["train", "unused"], default="train", help="Whether the four historical datasets go to train or are excluded/unused.")
    args = parser.parse_args()
    fractions = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]
    build_scenarios(args.base_dir, args.out_base, fractions, args.seed, args.prefix, args.image_mode, args.other_datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
