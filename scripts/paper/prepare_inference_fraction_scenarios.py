#!/usr/bin/env python3
"""Create nested, inference-only train-fraction datasets with fixed validation/test splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "data/otite_ds_224/USA_Turquie_Chili_GMFUNL_inference_20260803"
DEFAULT_OUT = ROOT / "data/otite_ds_224"


def label_for(frac: float) -> str:
    return (f"{frac:.3f}".rstrip("0").rstrip(".")).replace(".", "p")


def allocation(labels: pd.Series, n: int) -> dict[str, int]:
    counts = labels.astype(str).value_counts().sort_index().to_dict()
    total = sum(counts.values())
    if not 0 <= n <= total:
        raise ValueError(f"Requested {n} samples from {total}")
    raw = {key: n * value / total for key, value in counts.items()}
    out = {key: int(np.floor(value)) for key, value in raw.items()}
    for key in sorted(counts, key=lambda k: (raw[k] - out[k], counts[k], k), reverse=True)[: n - sum(out.values())]:
        out[key] += 1
    if n >= len(counts):
        for key in counts:
            if out[key] == 0:
                donor = max((k for k in counts if out[k] > 1), key=lambda k: (out[k], raw[k] - out[k], k))
                out[key], out[donor] = 1, out[donor] - 1
    return out


def stratified_orders(df: pd.DataFrame, seed: int) -> dict[str, list[int]]:
    """Return deterministic shuffled per-class queues for nested stratified subsets."""
    rng = np.random.default_rng(seed)
    return {
        label: list(rng.permutation(df[df["label"].astype(str).eq(label)].index.to_numpy()))
        for label in sorted(df["label"].astype(str).unique())
    }


def take_stratified(df: pd.DataFrame, n: int, seed: int) -> list[int]:
    wanted = allocation(df["label"], n)
    rng = np.random.default_rng(seed)
    out: list[int] = []
    for label in sorted(wanted):
        pool = df[df["label"].astype(str).eq(label)].index.to_numpy()
        out.extend(rng.choice(pool, size=wanted[label], replace=False).tolist())
    return out


def link_images(base: Path, target: Path, names: pd.Series, mode: str) -> None:
    for name in sorted(set(names.astype(str))):
        src, dst = base / name, target / name
        if dst.exists() or dst.is_symlink():
            continue
        if mode == "hardlink":
            os.link(src, dst)
        elif mode == "symlink":
            os.symlink(os.path.relpath(src, target), dst)
        else:
            shutil.copy2(src, dst)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    p.add_argument("--out-base", type=Path, default=DEFAULT_OUT)
    p.add_argument("--prefix", default="USA_Turquie_Chili_GMFUNL_inference_fraction_v1")
    p.add_argument("--fractions", default="0.5,0.25,0.1,0.05,0.02")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    p.add_argument("--historical-group", choices=("train", "unused"), default="unused", help="Assignment for all non-inference rows; validation and test are always inference-only.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    base = pd.read_csv(args.base_dir / "infos.csv")
    inf = base[base.dataset.eq("inference")].copy()
    if inf.empty:
        raise ValueError("No inference rows in base infos.csv")
    total = len(inf)
    # Deterministic fixed hold-outs: 66 validation and 65 test samples for 262 rows.
    # This is the closest integer implementation of 25%/25%, and 0.5 train is 131 exactly.
    valid_n = int(np.ceil(total * .25))
    test_n = int(np.floor(total * .25))
    valid_idx = take_stratified(inf, valid_n, args.seed + 2000)
    remaining = inf.drop(index=valid_idx)
    test_idx = take_stratified(remaining, test_n, args.seed + 1000)
    train_pool = inf.drop(index=valid_idx + test_idx)
    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
    if fractions != sorted(fractions, reverse=True):
        raise ValueError("Fractions must be supplied in descending execution order.")

    # One deterministic rank order makes smaller training samples strict subsets of larger ones.
    ordered_train = stratified_orders(train_pool, args.seed + 3000)
    rows = []
    for frac in fractions:
        n_train = int(round(total * frac))
        if n_train > len(train_pool):
            raise ValueError(f"train fraction {frac} needs {n_train}, only {len(train_pool)} remain")
        out_dir = args.out_base / f"{args.prefix}_train{label_for(frac)}_seed{args.seed}"
        if out_dir.exists() and any(out_dir.iterdir()):
            if not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {out_dir}; pass --overwrite")
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = base.copy()
        out["group"] = "unused"
        out.loc[~out.dataset.eq("inference"), "group"] = args.historical_group
        out.loc[test_idx, "group"] = "test"
        out.loc[valid_idx, "group"] = "valid"
        requested = allocation(train_pool["label"], n_train)
        train_idx = [idx for label in sorted(requested) for idx in ordered_train[label][:requested[label]]]
        out.loc[train_idx, "group"] = "train"
        out.to_csv(out_dir / "infos.csv", index=False)
        link_images(args.base_dir, out_dir, out["name"], args.image_mode)
        summary = {
            "scenario_fraction": frac, "scenario_label": label_for(frac), "seed": args.seed,
            "dataset_path": str(out_dir.relative_to(ROOT)), "total_inference": total,
            "inference_train": int((out.dataset.eq("inference") & out.group.eq("train")).sum()),
            "inference_valid": int((out.dataset.eq("inference") & out.group.eq("valid")).sum()),
            "inference_test": int((out.dataset.eq("inference") & out.group.eq("test")).sum()),
            "inference_unused": int((out.dataset.eq("inference") & out.group.eq("unused")).sum()),
            "historical_train": int((~out.dataset.eq("inference") & out.group.eq("train")).sum()),
            "historical_valid_test": int((~out.dataset.eq("inference") & out.group.isin(["valid", "test"])).sum()),
            "infos_sha256": hashlib.sha256((out_dir / "infos.csv").read_bytes()).hexdigest(),
        }
        (out_dir / "scenario_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        rows.append(summary)
    manifest = args.out_base / f"{args.prefix}_seed{args.seed}_scenarios.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote scenario manifest: {manifest}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
