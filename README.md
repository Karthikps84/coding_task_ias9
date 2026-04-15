# Deep Learning for Crystal Orientation Mapping in 4D-STEM

**Task:** Predict crystallographic orientation (Euler angles) from electron diffraction patterns.  
**Dataset:** LNO_simulated_test_dataset — 256×256 PNG diffraction images of LiNiO₂.

---

## Key Innovation: SO(3)-Native Learning

Rather than regressing Euler angles directly — which suffer from **periodicity, gimbal-lock, and discontinuities** — this solution predicts a **6D rotation representation** that maps *continuously* to SO(3), optimised with a **geodesic (angular) loss** directly on the rotation manifold.

### Why Euler angles are problematic for regression:

| Issue | Description |
|---|---|
| Periodicity | φ₁ and φ₂ wrap at 360°, causing large loss spikes near boundaries |
| Gimbal lock | Φ≈0° makes φ₁ and φ₂ degenerate |
| Discontinuities | Small rotational changes → large Euler angle jumps |
| Symmetry | Cubic/hexagonal crystal symmetry creates equivalent orientations |

### Solution: 6D Rotation Representation (Zhou et al., CVPR 2019)

1. **Output:** 6 numbers (first two columns of a 3×3 rotation matrix)
2. **Recovery:** Gram-Schmidt orthogonalisation → valid SO(3) element
3. **Loss:** Geodesic distance `d(R₁,R₂) = arccos((tr(R₁ᵀR₂)−1)/2)` in radians

This directly minimises the physically meaningful angular error, with no discontinuities.

---

## Architecture

```
Input: 256×256 diffraction PNG
  │
  ▼
EfficientNet-B2 backbone (ImageNet pretrained)
  │  — compound-scaled for optimal depth/width/resolution balance
  │  — rich low-level feature detector for spot patterns
  ▼
SE Attention block (channel recalibration)
  │  — reweights feature maps to focus on informative spots
  ▼
FC Head: 1408 → 512 → 256 → 6
  │  — LayerNorm + SiLU + Dropout
  ▼
6D output → Gram-Schmidt → R ∈ SO(3)
  │
  ▼
Geodesic Loss (optimise directly on rotation manifold)
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Data Exploration
```bash
python explore.py \
    --data_dir /path/to/LNO_simulated_test_dataset \
    --max_samples 50000 \
    --out_dir results
```
Generates:
- `explore_orientations.png` — angle distributions, 2D histograms, Rodrigues space
- `explore_patterns.png` — sample diffraction pattern grid
- `explore_pixel_stats.png` — pixel intensity statistics
- `explore_similar.png` — illustration of the core challenge

### Step 2: Training
```bash
# Quick test (CPU/small GPU, ~10 min):
python train.py \
    --data_dir /path/to/LNO_simulated_test_dataset \
    --max_samples 5000 \
    --epochs 15 \
    --batch_size 16

# Full run (GPU, ~1-2 hrs):
python train.py \
    --data_dir /path/to/LNO_simulated_test_dataset \
    --max_samples 50000 \
    --epochs 50 \
    --batch_size 64 \
    --num_workers 8
```

### Step 3: Evaluate only (load saved checkpoint)
```bash
python train.py \
    --data_dir /path/to/LNO_simulated_test_dataset \
    --eval_only
```

---

## Output Files

```
checkpoints/
  best_model.pt        ← best checkpoint by val MAE
  final_model.pt       ← end-of-training checkpoint
  history.json         ← loss and MAE per epoch

results/
  metrics.json                ← geodesic error, per-angle MAE, percentile stats
  training_curves.png         ← loss and MAE curves
  error_distribution.png      ← geodesic + per-angle error histograms
  pred_vs_true.png            ← scatter: predicted vs true Euler angles
  sample_predictions.png      ← grid of pattern + prediction
  best_cases.png              ← model's most accurate predictions
  worst_cases.png             ← failure cases for analysis
```

---

## Design Choices & Justification

| Choice | Rationale |
|---|---|
| **6D rotation repr.** | Continuous, avoids Euler periodicity/gimbal-lock |
| **Geodesic loss** | Optimises the true angular error metric |
| **EfficientNet-B2** | Strong backbone, well-regularised, ImageNet init |
| **SE Attention** | Focuses on informative diffraction spots |
| **MC-Dropout** | Free uncertainty estimate without ensemble overhead |
| **OneCycleLR → Cosine** | Faster convergence, better generalisation |
| **Augmentation** | Blur, brightness/contrast — simulates experimental variation |

---

## Evaluation Metrics

- **Mean Geodesic Error (°):** primary metric — average angular distance on SO(3)
- **Median Geodesic Error (°):** robust to outliers
- **% within 5°/10°/20°:** practical accuracy thresholds
- **Per-angle MAE (φ₁, Φ, φ₂):** diagnostic breakdown

---

## References

1. Scheunert et al. (2026). *Determining the grain orientations of battery materials from electron diffraction patterns using CNNs.* npj Computational Materials.
2. Zhou et al. (2019). *On the Continuity of Rotation Representations in Neural Networks.* CVPR.
