"""
Step 1 — Data Exploration
=========================
Visualise diffraction patterns, analyse orientation distributions,
and identify potential challenges before model design.

Run:
    python explore.py --data_dir /path/to/LNO_simulated_test_dataset
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# ─────────────────────────────────────────────
_FNAME_RE = re.compile(
    r"phi1_(?P<phi1>[\d.]+)_phi_(?P<phi>[\d.]+)_phi2_(?P<phi2>[\d.]+)"
)


def load_metadata(data_dir: str, max_samples: int = 5000) -> pd.DataFrame:
    """Parse filenames to extract Euler angles."""
    records = []
    for fp in sorted(Path(data_dir).glob("*.png")):
        m = _FNAME_RE.search(fp.name)
        if m:
            records.append({
                "path": str(fp),
                "phi1": float(m.group("phi1")),
                "phi":  float(m.group("phi")),
                "phi2": float(m.group("phi2")),
            })
    df = pd.DataFrame(records)
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42).reset_index(drop=True)
    print(f"[Explore] Loaded metadata for {len(df)} images")
    return df


def plot_orientation_distribution(df: pd.DataFrame, save_path: str = "results/explore_orientations.png"):
    """Visualise the distribution of Euler angles in orientation space."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    angles = [("phi1", "φ₁ (°)", "#2563eb"),
               ("phi",  "Φ  (°)",  "#16a34a"),
               ("phi2", "φ₂ (°)", "#dc2626")]

    # 1D histograms
    for i, (col, label, color) in enumerate(angles):
        ax = fig.add_subplot(gs[0, i])
        ax.hist(df[col], bins=60, color=color, edgecolor="white", alpha=0.85)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"Distribution of {label}", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
        stats = df[col].describe()
        ax.text(0.97, 0.95, f"μ={stats['mean']:.1f}°\nσ={stats['std']:.1f}°",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))

    # Pairwise scatter (phi1 vs phi2 coloured by phi)
    ax_scatter = fig.add_subplot(gs[0, 3])
    sc = ax_scatter.scatter(df["phi1"], df["phi2"], c=df["phi"],
                             cmap="viridis", s=3, alpha=0.4, rasterized=True)
    plt.colorbar(sc, ax=ax_scatter, label="Φ (°)")
    ax_scatter.set_xlabel("φ₁ (°)"); ax_scatter.set_ylabel("φ₂ (°)")
    ax_scatter.set_title("φ₁ vs φ₂ (colour=Φ)", fontweight="bold")
    ax_scatter.grid(alpha=0.3)

    # 2D histograms
    pairs = [("phi1", "phi"), ("phi", "phi2"), ("phi1", "phi2")]
    labels_map = {"phi1": "φ₁ (°)", "phi": "Φ (°)", "phi2": "φ₂ (°)"}
    for i, (a, b) in enumerate(pairs):
        ax = fig.add_subplot(gs[1, i])
        h = ax.hist2d(df[a], df[b], bins=40, cmap="inferno")
        plt.colorbar(h[3], ax=ax, label="Count")
        ax.set_xlabel(labels_map[a]); ax.set_ylabel(labels_map[b])
        ax.set_title(f"{labels_map[a]} vs {labels_map[b]}", fontweight="bold", fontsize=9)

    # Fundamental zone coverage — rodrigues space approximation
    ax_rod = fig.add_subplot(gs[1, 3], projection="3d")
    r = Rotation.from_euler("ZXZ", df[["phi1", "phi", "phi2"]].values, degrees=True)
    rod = r.as_rotvec()  # approximation of Rodrigues space
    ax_rod.scatter(rod[:, 0], rod[:, 1], rod[:, 2], s=1, alpha=0.2,
                   c=df["phi"].values, cmap="plasma", rasterized=True)
    ax_rod.set_title("Rotation vector space coverage", fontweight="bold", fontsize=8)
    ax_rod.set_xlabel("r₀"); ax_rod.set_ylabel("r₁"); ax_rod.set_zlabel("r₂")

    plt.suptitle("Dataset Orientation Distribution — LNO Simulated", fontsize=13, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {save_path}")


def plot_sample_patterns(df: pd.DataFrame, n: int = 16, save_path: str = "results/explore_patterns.png"):
    """Show a grid of sample diffraction patterns with their labels."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sample = df.sample(n, random_state=7).reset_index(drop=True)
    cols = 4; rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = axes.flatten()

    for i, row in sample.iterrows():
        img = np.array(Image.open(row["path"]).convert("L"))
        axes[i].imshow(img, cmap="inferno")
        axes[i].set_title(
            f"φ₁={row.phi1:.1f}°\nΦ={row.phi:.1f}°  φ₂={row.phi2:.1f}°",
            fontsize=7.5
        )
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Sample Diffraction Patterns — LNO Simulated", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {save_path}")


def plot_similar_orientations(df: pd.DataFrame, save_path: str = "results/explore_similar.png"):
    """
    Highlight the core challenge: patterns with very similar orientations
    may look different, and patterns with different orientations may look similar.
    Show pairs of (close, distant) orientation images.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Find pairs close in orientation space
    sample = df.sample(300, random_state=1).reset_index(drop=True)
    r = Rotation.from_euler("ZXZ", sample[["phi1", "phi", "phi2"]].values, degrees=True)
    mats = r.as_matrix()

    close_pairs, far_pairs = [], []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            M = mats[i].T @ mats[j]
            angle = np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1)) * 180 / np.pi
            if angle < 3 and len(close_pairs) < 3:
                close_pairs.append((i, j, angle))
            if angle > 60 and len(far_pairs) < 3:
                far_pairs.append((i, j, angle))
        if len(close_pairs) >= 3 and len(far_pairs) >= 3:
            break

    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    row_labels = [f"Δ={a:.1f}°" for _, _, a in close_pairs]

    for row_i, (i, j, angle) in enumerate(close_pairs):
        for col_i, idx in enumerate([i, j]):
            img = np.array(Image.open(sample.iloc[idx]["path"]).convert("L"))
            ax = axes[row_i][col_i]
            ax.imshow(img, cmap="inferno")
            r2 = sample.iloc[idx]
            ax.set_title(f"φ₁={r2.phi1:.1f}° Φ={r2.phi:.1f}° φ₂={r2.phi2:.1f}°", fontsize=7)
            ax.axis("off")
        axes[row_i][0].set_ylabel(f"CLOSE Δ={angle:.1f}°", fontsize=8, color="green")

    for row_i, (i, j, angle) in enumerate(far_pairs):
        for col_i, idx in enumerate([i, j]):
            img = np.array(Image.open(sample.iloc[idx]["path"]).convert("L"))
            ax = axes[row_i][2 + col_i]
            ax.imshow(img, cmap="inferno")
            r2 = sample.iloc[idx]
            ax.set_title(f"φ₁={r2.phi1:.1f}° Φ={r2.phi:.1f}° φ₂={r2.phi2:.1f}°", fontsize=7)
            ax.axis("off")

    axes[0][0].set_ylabel("CLOSE\npair", fontsize=8, color="green", fontweight="bold")
    axes[0][2].set_ylabel("DISTANT\npair", fontsize=8, color="red", fontweight="bold")

    plt.suptitle("Key Challenge: Patterns with Similar vs. Distant Orientations",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {save_path}")


def plot_pixel_statistics(df: pd.DataFrame, n: int = 200, save_path: str = "results/explore_pixel_stats.png"):
    """Analyse image statistics: mean intensity, std, dynamic range."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sample = df.sample(min(n, len(df)), random_state=0)

    means, stds, maxes = [], [], []
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Computing pixel stats"):
        img = np.array(Image.open(row["path"]).convert("L"), dtype=np.float32)
        means.append(img.mean())
        stds.append(img.std())
        maxes.append(img.max())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, vals, label, color in zip(
        axes,
        [means, stds, maxes],
        ["Mean pixel intensity", "Pixel std", "Max pixel value"],
        ["#6366f1", "#f59e0b", "#10b981"]
    ):
        ax.hist(vals, bins=40, color=color, edgecolor="white", alpha=0.85)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(label, fontweight="bold")
        ax.axvline(np.mean(vals), color="red", ls="--", label=f"Mean={np.mean(vals):.1f}")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Pixel-level Statistics Across Diffraction Patterns", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--out_dir",     type=str, default="results")
    args = parser.parse_args()

    df = load_metadata(args.data_dir, args.max_samples)

    # Summary statistics
    print("\n── Euler Angle Statistics ──")
    print(df[["phi1", "phi", "phi2"]].describe().round(2).to_string())

    print("\n── Image count:", len(df), "──")

    plot_orientation_distribution(df, save_path=f"{args.out_dir}/explore_orientations.png")
    plot_sample_patterns(df, save_path=f"{args.out_dir}/explore_patterns.png")
    plot_pixel_statistics(df, save_path=f"{args.out_dir}/explore_pixel_stats.png")

    try:
        plot_similar_orientations(df, save_path=f"{args.out_dir}/explore_similar.png")
    except Exception as e:
        print(f"[WARN] Similar orientation plot skipped: {e}")

    print(f"\n[Done] All exploration plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
