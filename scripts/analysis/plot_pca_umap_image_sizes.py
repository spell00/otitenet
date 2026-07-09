from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def find_image_path(root: Path, dataset: str, name: str) -> Path | None:
    candidates = [
        root / dataset / name,
        root / "images" / dataset / name,
        root / "imgs" / dataset / name,
        root / name,
    ]

    for p in candidates:
        if p.exists():
            return p

    matches = list(root.rglob(name))
    if matches:
        return matches[0]

    return None


def load_image_vector(path: Path, feature_size: int | None) -> np.ndarray:
    img = Image.open(path).convert("RGB")

    if feature_size is not None:
        img = img.resize((feature_size, feature_size), Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def make_scatter(
    coords: np.ndarray,
    meta: pd.DataFrame,
    color_col: str,
    title: str,
    out_path: Path,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    values = meta[color_col].astype(str).fillna("NA")
    labels = sorted(values.unique())

    for label in labels:
        mask = values == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.75,
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(
        title=color_col,
        bbox_to_anchor=(1.04, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_one_size(
    root: Path,
    size_label: str,
    feature_size: int | None,
    out_dir: Path,
    max_samples: int | None,
    random_state: int,
    color_cols: list[str],
    run_umap: bool,
) -> None:
    infos_path = root / "infos.csv"
    if not infos_path.exists():
        print(f"[SKIP] Missing infos.csv: {infos_path}")
        return

    df = pd.read_csv(infos_path)

    required = {"dataset", "name", "label"}
    missing = required - set(df.columns)
    if missing:
        print(f"[SKIP] {infos_path} missing columns: {sorted(missing)}")
        return

    df = df.copy()
    df["image_path"] = [
        find_image_path(root, str(dataset), str(name))
        for dataset, name in zip(df["dataset"], df["name"])
    ]

    missing_images = df["image_path"].isna().sum()
    if missing_images:
        print(f"[WARN] {size_label}: {missing_images} images not found; they will be skipped.")

    df = df[df["image_path"].notna()].reset_index(drop=True)

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(max_samples, random_state=random_state).reset_index(drop=True)

    if len(df) < 3:
        print(f"[SKIP] {size_label}: not enough images after filtering.")
        return

    print(f"[LOAD] {size_label}: loading {len(df)} images from {root}")

    X = np.stack(
        [
            load_image_vector(Path(p), feature_size=feature_size)
            for p in df["image_path"]
        ],
        axis=0,
    )

    print(f"[FIT] {size_label}: feature matrix shape = {X.shape}")

    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_ * 100
    print(
        f"[PCA] {size_label}: PC1={explained[0]:.2f}%, "
        f"PC2={explained[1]:.2f}%"
    )

    size_out = out_dir / f"size_{size_label}"
    size_out.mkdir(parents=True, exist_ok=True)

    for color_col in color_cols:
        if color_col not in df.columns:
            print(f"[WARN] {size_label}: missing column {color_col!r}; skipping.")
            continue

        make_scatter(
            pca_coords,
            df,
            color_col=color_col,
            title=f"PCA - otite_ds_{size_label} colored by {color_col}",
            out_path=size_out / f"pca_by_{color_col}.png",
            xlabel=f"PC1 ({explained[0]:.1f}%)",
            ylabel=f"PC2 ({explained[1]:.1f}%)",
        )

    if run_umap:
        try:
            import umap
        except ImportError:
            print("[WARN] umap-learn is not installed. Install with: pip install umap-learn")
            return

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=random_state,
        )
        umap_coords = reducer.fit_transform(X)

        for color_col in color_cols:
            if color_col not in df.columns:
                continue

            make_scatter(
                umap_coords,
                df,
                color_col=color_col,
                title=f"UMAP - otite_ds_{size_label} colored by {color_col}",
                out_path=size_out / f"umap_by_{color_col}.png",
                xlabel="UMAP 1",
                ylabel="UMAP 2",
            )

    df_out = df.drop(columns=["image_path"]).copy()
    df_out["pca_1"] = pca_coords[:, 0]
    df_out["pca_2"] = pca_coords[:, 1]
    df_out.to_csv(size_out / "plot_metadata_with_pca.csv", index=False)

    print(f"[DONE] {size_label}: wrote plots to {size_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PCA and UMAP plots from otitenet image datasets at multiple image sizes."
    )
    parser.add_argument(
        "--base-dir",
        default="/home/simon/otitenet/data",
        help="Base data directory containing otite_ds_-1, otite_ds_32, otite_ds_64, etc.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["-1", "32", "64", "224"],
        help="Dataset size suffixes to process.",
    )
    parser.add_argument(
        "--feature-size",
        default="auto",
        help=(
            "Image size used for PCA/UMAP features. "
            "Use 'auto' to use the dataset size, except -1 uses 224. "
            "Use an integer like 64 to force all images to that size."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="/home/simon/otitenet/output/analysis/image_size_pca_umap",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional subsampling for faster testing, e.g. --max-samples 1000.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--color-cols",
        nargs="+",
        default=["label", "dataset"],
        help="Columns to color plots by. group is ignored unless you explicitly pass it.",
    )
    parser.add_argument(
        "--no-umap",
        action="store_true",
        help="Only run PCA.",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for size_label in args.sizes:
        root = base_dir / f"otite_ds_{size_label}" / "USA_Turquie_Chili_GMFUNL_inference"

        if args.feature_size == "auto":
            if size_label == "-1":
                feature_size = 224
            else:
                feature_size = int(size_label)
        else:
            feature_size = int(args.feature_size)

        run_one_size(
            root=root,
            size_label=size_label,
            feature_size=feature_size,
            out_dir=out_dir,
            max_samples=args.max_samples,
            random_state=args.random_state,
            color_cols=args.color_cols,
            run_umap=not args.no_umap,
        )


if __name__ == "__main__":
    main()