"""
4D-STEM Crystal Orientation Mapping via Deep Learning
======================================================
Key Innovation: Instead of regressing Euler angles directly (which suffer from
periodicity, gimbal-lock, and discontinuities), we predict a 6D rotation
representation that maps continuously to SO(3), supervised with a geodesic
(angular) loss on the rotation manifold.

Reference: Zhou et al. (2019) "On the Continuity of Rotation Representations in
Neural Networks" — CVPR 2019.

Architecture: EfficientNet-B2 backbone + SE-attention + dual-head output
(6D rotation repr. + optional uncertainty via MC-Dropout).
"""

import os
import re
import time
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torchvision import transforms, models
import torchvision.transforms.functional as TF

from PIL import Image
from scipy.spatial.transform import Rotation

# ─────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════
#  ROTATION MATH  (SO(3) operations)
# ═══════════════════════════════════════════════════════════════════════

def euler_to_rotmat(phi1, phi, phi2):
    """
    Bunge convention (ZXZ): φ1 → Φ → φ2  (all in radians).
    Returns rotation matrix R ∈ SO(3).
    """
    r = Rotation.from_euler('ZXZ', np.stack([phi1, phi, phi2], axis=-1))
    return r.as_matrix()  # (..., 3, 3)


def rotmat_to_6d(R):
    """
    Projects a 3×3 rotation matrix to its 6D continuous representation
    (first two columns). Shape: (..., 6).
    """
    return R[..., :, :2].reshape(*R.shape[:-2], 6)   # columns 0 and 1


def sixd_to_rotmat(x):
    """
    Recovers a valid SO(3) matrix from a 6D vector via Gram-Schmidt.
    Input shape: (..., 6)  →  Output shape: (..., 3, 3)
    """
    a1 = x[..., :3]
    a2 = x[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)          # (..., 3, 3)


def rotmat_to_euler(R):
    """
    SO(3) → Bunge Euler angles (degrees).
    Uses scipy for numerical stability.
    """
    if isinstance(R, torch.Tensor):
        R = R.detach().cpu().numpy()
    r = Rotation.from_matrix(R)
    return r.as_euler('ZXZ', degrees=True)             # (..., 3)


def geodesic_distance(R1, R2):
    """
    Angular distance (radians) between two rotation matrices.
    d(R1, R2) = || log(R1^T R2) ||_F / sqrt(2)
    """
    M = torch.bmm(R1.transpose(-1, -2), R2)
    # clamp for numerical stability of acos
    trace = M.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_angle = ((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos_angle)   # radians


def mean_angular_error_deg(R_pred, R_true):
    """Returns mean angular error in degrees."""
    err_rad = geodesic_distance(R_pred, R_true)
    return err_rad.mean().item() * 180 / np.pi


# ═══════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════

# Regex to parse Euler angles from filename
_FNAME_RE = re.compile(
    r"phi1_(?P<phi1>[\d.]+)_phi_(?P<phi>[\d.]+)_phi2_(?P<phi2>[\d.]+)"
)

class DiffractionDataset(Dataset):
    """
    Loads diffraction pattern PNGs from a directory.
    Ground-truth Euler angles are parsed from filenames.
    Returns image tensor + 6D rotation vector (continuous SO(3) target).
    """

    def __init__(self, image_dir: str, max_samples: int = None, augment: bool = True):
        self.image_dir = Path(image_dir)
        self.augment = augment

        # Collect all PNG files and parse orientations
        records = []
        for fp in sorted(self.image_dir.glob("*.png")):
            m = _FNAME_RE.search(fp.name)
            if m:
                records.append({
                    "path": fp,
                    "phi1": float(m.group("phi1")),
                    "phi":  float(m.group("phi")),
                    "phi2": float(m.group("phi2")),
                })

        if not records:
            raise FileNotFoundError(f"No PNG files found in {image_dir}")

        self.df = pd.DataFrame(records)

        # Sub-sample if requested
        if max_samples and max_samples < len(self.df):
            self.df = self.df.sample(max_samples, random_state=42).reset_index(drop=True)

        print(f"[Dataset] Loaded {len(self.df)} samples from {image_dir}")

        # Pre-compute rotation matrices and 6D targets (numpy)
        phi1_rad = np.deg2rad(self.df["phi1"].values)
        phi_rad  = np.deg2rad(self.df["phi"].values)
        phi2_rad = np.deg2rad(self.df["phi2"].values)
        R = euler_to_rotmat(phi1_rad, phi_rad, phi2_rad)  # (N, 3, 3)
        # 6D: first two columns, flattened
        self.targets_6d = R[:, :, :2].reshape(-1, 6).astype(np.float32)  # (N, 6)
        self.rotmats    = R.astype(np.float32)

        # Transforms
        self.base_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.aug_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("L")

        tf = self.aug_tf if self.augment else self.base_tf
        img_t = tf(img)                              # (1, 224, 224)
        # Expand to 3 channels (EfficientNet expects RGB)
        img_t = img_t.expand(3, -1, -1)

        target_6d = torch.tensor(self.targets_6d[idx], dtype=torch.float32)
        rotmat    = torch.tensor(self.rotmats[idx],    dtype=torch.float32)

        return img_t, target_6d, rotmat


# ═══════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════

class SqueezeExcitation(nn.Module):
    """Channel-wise SE block for feature recalibration."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x).view(-1, x.size(1), 1, 1)
        return x * scale


class OrientationNet(nn.Module):
    """
    EfficientNet-B2 backbone + SE-attention head → 6D rotation output.

    Why EfficientNet?
      - Strong inductive bias for image features
      - Compound scaling balances depth/width/resolution
      - Pre-trained on ImageNet → rich low-level feature detector

    Why 6D output?
      - Continuous mapping to SO(3) (no discontinuities unlike Euler angles)
      - Gram-Schmidt orthogonalisation recovers valid rotation matrix
      - Geodesic loss directly optimises angular error on the manifold

    Optional: MC-Dropout for epistemic uncertainty estimation.
    """

    def __init__(self, dropout_p: float = 0.3, use_pretrained: bool = True):
        super().__init__()

        weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if use_pretrained else None
        backbone = models.efficientnet_b2(weights=weights)

        # Remove classifier; keep feature extractor
        self.features = backbone.features  # output: (B, 1408, 7, 7)
        self.pool     = nn.AdaptiveAvgPool2d(1)

        feat_dim = 1408  # EfficientNet-B2 feature dim
        self.se_attention = SqueezeExcitation(feat_dim, reduction=16)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 6),   # 6D rotation output
        )

    def forward(self, x):
        f = self.features(x)           # (B, 1408, 7, 7)
        f = self.se_attention(f)       # channel recalibration
        f = self.pool(f)               # (B, 1408, 1, 1)
        out_6d = self.head(f)          # (B, 6)
        return out_6d

    def predict_rotmat(self, x):
        """Convenience: forward + Gram-Schmidt → SO(3)."""
        out_6d = self.forward(x)
        return sixd_to_rotmat(out_6d)


# ═══════════════════════════════════════════════════════════════════════
#  LOSS
# ═══════════════════════════════════════════════════════════════════════

class GeodesicLoss(nn.Module):
    """
    Loss = mean geodesic distance on SO(3).
    This directly minimises the angular error — the physically meaningful metric.
    """
    def forward(self, pred_6d: torch.Tensor, target_rotmat: torch.Tensor) -> torch.Tensor:
        R_pred = sixd_to_rotmat(pred_6d)
        loss   = geodesic_distance(R_pred, target_rotmat)
        return loss.mean()


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model        = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.criterion    = GeodesicLoss()
        self.optimizer    = torch.optim.AdamW(
            model.parameters(), lr=config["lr"], weight_decay=1e-4
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config["epochs"], eta_min=1e-6
        )
        self.history = {"train_loss": [], "val_loss": [], "val_mae_deg": []}
        self.best_val_mae = float("inf")

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for imgs, targets_6d, rotmats in tqdm(self.train_loader, desc="  Train", leave=False):
            imgs      = imgs.to(DEVICE)
            targets_6d = targets_6d.to(DEVICE)
            rotmats   = rotmats.to(DEVICE)

            self.optimizer.zero_grad()
            pred_6d = self.model(imgs)
            loss    = self.criterion(pred_6d, rotmats)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def eval_epoch(self):
        self.model.eval()
        total_loss, total_mae = 0, 0
        for imgs, targets_6d, rotmats in tqdm(self.val_loader, desc="  Val  ", leave=False):
            imgs       = imgs.to(DEVICE)
            rotmats    = rotmats.to(DEVICE)
            pred_6d    = self.model(imgs)
            loss       = self.criterion(pred_6d, rotmats)
            R_pred     = sixd_to_rotmat(pred_6d)
            mae_deg    = mean_angular_error_deg(R_pred, rotmats)
            total_loss += loss.item()
            total_mae  += mae_deg
        return total_loss / len(self.val_loader), total_mae / len(self.val_loader)

    def run(self, save_dir: str = "checkpoints"):
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n{'─'*60}")
        print(f"  Training for {self.config['epochs']} epochs | device={DEVICE}")
        print(f"{'─'*60}")

        for epoch in range(1, self.config["epochs"] + 1):
            t0 = time.time()
            train_loss          = self.train_epoch()
            val_loss, val_mae   = self.eval_epoch()
            self.scheduler.step()
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae_deg"].append(val_mae)

            print(
                f"  Epoch {epoch:03d}/{self.config['epochs']} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_MAE={val_mae:.2f}° | {elapsed:.1f}s"
            )

            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                ckpt_path = os.path.join(save_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "val_mae_deg": val_mae,
                    "config": self.config,
                }, ckpt_path)
                print(f"    ✓ Saved best model (val_MAE={val_mae:.2f}°) → {ckpt_path}")

        # Save final
        torch.save({
            "epoch": self.config["epochs"],
            "model_state": self.model.state_dict(),
            "history": self.history,
            "config": self.config,
        }, os.path.join(save_dir, "final_model.pt"))

        # Save training history
        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\n  Best val MAE: {self.best_val_mae:.2f}°")
        return self.history


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION & VISUALISATION
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model, loader, save_dir="results"):
    """Full evaluation: per-angle errors, geodesic error distribution, failure cases."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    all_pred_euler, all_true_euler, all_geo_err = [], [], []
    sample_imgs, sample_pred, sample_true = [], [], []

    for imgs, _, rotmats in tqdm(loader, desc="Evaluating"):
        imgs    = imgs.to(DEVICE)
        rotmats = rotmats.to(DEVICE)

        pred_6d  = model(imgs)
        R_pred   = sixd_to_rotmat(pred_6d)
        geo_err  = geodesic_distance(R_pred, rotmats).cpu().numpy() * 180 / np.pi

        pred_euler = rotmat_to_euler(R_pred.cpu().numpy())
        true_euler = rotmat_to_euler(rotmats.cpu().numpy())

        all_pred_euler.append(pred_euler)
        all_true_euler.append(true_euler)
        all_geo_err.append(geo_err)

        # Save a few samples for visualisation
        if len(sample_imgs) < 12:
            for i in range(min(imgs.size(0), 12 - len(sample_imgs))):
                sample_imgs.append(imgs[i].cpu())
                sample_pred.append(pred_euler[i])
                sample_true.append(true_euler[i])

    all_pred_euler = np.concatenate(all_pred_euler, axis=0)
    all_true_euler = np.concatenate(all_true_euler, axis=0)
    all_geo_err    = np.concatenate(all_geo_err, axis=0)

    metrics = {
        "mean_geodesic_error_deg": float(all_geo_err.mean()),
        "median_geodesic_error_deg": float(np.median(all_geo_err)),
        "std_geodesic_error_deg": float(all_geo_err.std()),
        "pct_within_5deg": float((all_geo_err < 5).mean() * 100),
        "pct_within_10deg": float((all_geo_err < 10).mean() * 100),
        "pct_within_20deg": float((all_geo_err < 20).mean() * 100),
        "phi1_MAE": float(np.abs(all_pred_euler[:, 0] - all_true_euler[:, 0]).mean()),
        "phi_MAE":  float(np.abs(all_pred_euler[:, 1] - all_true_euler[:, 1]).mean()),
        "phi2_MAE": float(np.abs(all_pred_euler[:, 2] - all_true_euler[:, 2]).mean()),
    }

    # Print metrics
    print(f"\n{'═'*55}")
    print("  EVALUATION METRICS")
    print(f"{'═'*55}")
    for k, v in metrics.items():
        print(f"  {k:35s}: {v:.3f}")
    print(f"{'═'*55}")

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Plot 1: Training curves (if history exists) ──────────────────
    history_path = "checkpoints/history.json"
    if os.path.exists(history_path):
        with open(history_path) as f:
            hist = json.load(f)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(hist["train_loss"], label="Train loss", color="#2563eb")
        axes[0].plot(hist["val_loss"],   label="Val loss",   color="#dc2626")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Geodesic Loss (rad)")
        axes[0].set_title("Training vs Validation Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(hist["val_mae_deg"], color="#16a34a", label="Val MAE (°)")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Mean Angular Error (°)")
        axes[1].set_title("Validation MAE")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Plot] Saved training_curves.png")

    # ── Plot 2: Geodesic error distribution ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(all_geo_err, bins=80, color="#6366f1", edgecolor="white", alpha=0.85)
    axes[0].axvline(all_geo_err.mean(),   color="#dc2626", ls="--", lw=2, label=f"Mean={all_geo_err.mean():.1f}°")
    axes[0].axvline(np.median(all_geo_err), color="#f97316", ls="--", lw=2, label=f"Median={np.median(all_geo_err):.1f}°")
    axes[0].set_xlabel("Geodesic Error (°)"); axes[0].set_ylabel("Count")
    axes[0].set_title("Geodesic Error Distribution")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    labels = ["φ₁", "Φ", "φ₂"]
    colors = ["#2563eb", "#16a34a", "#dc2626"]
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        err = np.abs(all_pred_euler[:, i] - all_true_euler[:, i])
        axes[1].hist(err, bins=60, alpha=0.6, label=lbl, color=col)
    axes[1].set_xlabel("Absolute Error (°)"); axes[1].set_ylabel("Count")
    axes[1].set_title("Per-Euler-Angle Error Distribution")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved error_distribution.png")

    # ── Plot 3: Predicted vs. True scatter ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    angle_names = ["φ₁  (Bunge Z)", "Φ  (Bunge X)", "φ₂  (Bunge Z)"]
    for i, (ax, name) in enumerate(zip(axes, angle_names)):
        ax.scatter(all_true_euler[:, i], all_pred_euler[:, i],
                   s=2, alpha=0.3, color="#6366f1", rasterized=True)
        lo = min(all_true_euler[:, i].min(), all_pred_euler[:, i].min())
        hi = max(all_true_euler[:, i].max(), all_pred_euler[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect")
        ax.set_xlabel(f"True {name} (°)"); ax.set_ylabel(f"Pred {name} (°)")
        ax.set_title(f"{name}  MAE={np.abs(all_pred_euler[:,i]-all_true_euler[:,i]).mean():.2f}°")
        ax.legend(markerscale=3); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pred_vs_true.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved pred_vs_true.png")

    # ── Plot 4: Sample predictions grid ──────────────────────────────
    n = min(12, len(sample_imgs))
    cols = 4; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()
    for i in range(n):
        img_np = sample_imgs[i][0].numpy()   # grayscale channel
        img_np = (img_np * 0.5 + 0.5)        # un-normalize
        axes[i].imshow(img_np, cmap="inferno", vmin=0, vmax=1)
        p, t = sample_pred[i], sample_true[i]
        geo = np.arccos(np.clip(
            (np.trace(euler_to_rotmat(*np.deg2rad(t)).T @
                      euler_to_rotmat(*np.deg2rad(p))) - 1) / 2,
            -1, 1)) * 180 / np.pi
        axes[i].set_title(
            f"Pred: {p[0]:.1f}° {p[1]:.1f}° {p[2]:.1f}°\n"
            f"True: {t[0]:.1f}° {t[1]:.1f}° {t[2]:.1f}°\n"
            f"Δ={geo:.1f}°",
            fontsize=7, pad=3
        )
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Sample Predictions (φ₁, Φ, φ₂)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sample_predictions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved sample_predictions.png")

    # ── Plot 5: Failure analysis — worst predictions ──────────────────
    worst_idx = np.argsort(all_geo_err)[-8:]
    best_idx  = np.argsort(all_geo_err)[:8]
    for label, idxs in [("worst", worst_idx), ("best", best_idx)]:
        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        axes = axes.flatten()
        for ax_i, idx in enumerate(idxs):
            img_np = sample_imgs[min(idx, len(sample_imgs)-1)][0].numpy()
            img_np = img_np * 0.5 + 0.5
            axes[ax_i].imshow(img_np, cmap="inferno")
            axes[ax_i].set_title(f"Δ={all_geo_err[idx]:.1f}°", fontsize=8)
            axes[ax_i].axis("off")
        plt.suptitle(f"{label.capitalize()} predictions (geodesic error)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{label}_cases.png"), dpi=150, bbox_inches="tight")
        plt.close()
    print(f"  [Plot] Saved best/worst_cases.png")

    return metrics


# ═══════════════════════════════════════════════════════════════════════
#  MC-DROPOUT UNCERTAINTY
# ═══════════════════════════════════════════════════════════════════════

def mc_dropout_predict(model, imgs, n_passes: int = 20):
    """
    Estimate epistemic uncertainty via MC-Dropout.
    Keeps dropout active during inference for T forward passes.
    Returns: mean rotation matrix, std of geodesic distances (uncertainty).
    """
    model.train()  # activate dropout
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            pred_6d = model(imgs)
            R = sixd_to_rotmat(pred_6d)
            preds.append(R)

    model.eval()
    preds = torch.stack(preds, dim=0)   # (T, B, 3, 3)

    # Mean rotation via average of geodesic distances (approximate)
    mean_R = preds.mean(dim=0)
    # Normalise to SO(3) via SVD
    U, S, Vt = torch.linalg.svd(mean_R)
    mean_R   = U @ Vt

    # Uncertainty = std of geodesic distance from mean
    geo_stds = []
    for t in range(n_passes):
        d = geodesic_distance(preds[t], mean_R)   # (B,)
        geo_stds.append(d)
    uncertainty = torch.stack(geo_stds, dim=0).std(dim=0) * 180 / np.pi  # degrees

    return mean_R, uncertainty


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="4D-STEM Orientation Mapping")
    p.add_argument("--data_dir",    type=str, required=True,  help="Directory with PNG diffraction patterns")
    p.add_argument("--max_samples", type=int, default=10000,  help="Max number of images to use (default 10000)")
    p.add_argument("--epochs",      type=int, default=30,     help="Training epochs")
    p.add_argument("--batch_size",  type=int, default=32,     help="Batch size")
    p.add_argument("--lr",          type=float, default=3e-4, help="Learning rate")
    p.add_argument("--val_split",   type=float, default=0.10, help="Validation fraction")
    p.add_argument("--num_workers", type=int, default=8,      help="DataLoader workers")
    p.add_argument("--dropout",     type=float, default=0.3,  help="Dropout probability")
    p.add_argument("--eval_only",   action="store_true",      help="Skip training, load best_model.pt and evaluate")
    p.add_argument("--ckpt_dir",    type=str, default="checkpoints")
    p.add_argument("--results_dir", type=str, default="results")
    return p.parse_args()


def main():
    args = parse_args()
    config = {
        "epochs":      args.epochs,
        "batch_size":  args.batch_size,
        "lr":          args.lr,
        "max_samples": args.max_samples,
        "dropout":     args.dropout,
    }

    # ── Dataset ──────────────────────────────────────────────────────
    dataset = DiffractionDataset(
        image_dir   = args.data_dir,
        max_samples = args.max_samples,
        augment     = not args.eval_only,
    )

    total_size = len(dataset)

    n_train = int(0.8 * total_size)
    n_val   = int(0.1 * total_size)
    n_test  = total_size - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Disable augmentation on val and test splits for consistent evaluation
    val_ds.dataset.augment = False
    test_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test} samples")

    # ── Model ────────────────────────────────────────────────────────
    model = OrientationNet(dropout_p=args.dropout, use_pretrained=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    if args.eval_only:
        ckpt = torch.load(os.path.join(args.ckpt_dir, "best_model.pt"), map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded checkpoint (epoch {ckpt['epoch']}, val_MAE={ckpt['val_mae_deg']:.2f}°)")
    else:
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.run(save_dir=args.ckpt_dir)

    # ── Evaluation ───────────────────────────────────────────────────
    evaluate_model(model, test_loader, save_dir=args.results_dir)

    print(f"\n  All outputs saved to {args.results_dir}/")


if __name__ == "__main__":
    main()