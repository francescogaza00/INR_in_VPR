# INR in VPR

**Implicit Neural Representation in Visual Place Recognition**
*Thesis project — Politecnico di Torino.*

---

## Overview

This repository implements a geo-localisation pipeline that combines frozen visual-place-recognition (VPR) backbones with a lightweight MLP head to predict the camera position and bearing from a single RGB image.

The geographic area of interest is a small region in San Francisco (bounding box defined in Constants.py). To simplify the task,the spatial extent can be discretised into an M × M grid of cells (M); together with 12 bearing bins (30° each), this yields M² × 12 = total classes.

---

## Architecture

```
RGB image
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Frozen backbone                                 │
│  ┌──────────────────┐  ┌────────────────────┐    │
│  │    CosPlace      │  │      MegaLoc       │    │
│  │  ResNet-101      │  │  DINOv2 + SALAD    │    │
│  │  fc_output=2048  │  │  fc_output=8448    │    │
│  └──────────────────┘  └────────────────────┘    │
└──────────────────────────────────────────────────┘
    │
    ▼  descriptor vector (2048 or 8448-dim)
┌──────────────────────────────────────────────────┐
│  MLP head  (3 hidden layers, SiLU activations)   │
│  ┌───────────────┐  ┌──────────────────────────┐ │
│  │ Classifier    │  │      Regressor           │ │
│  │ M² × 12 logits│  |  SCALED / SPHERE3D /     │ │
│  │ (SOFT / HARD) │  │  HYPERSPHERE             │ │
│  └───────────────┘  └──────────────────────────┘ │
└──────────────────────────────────────────────────┘
    │
    ▼
   Loss (CrossEntropy or MSE)
```

---

## Label Modes

`GeoDataset` (`Get_dataset.py`) supports three label representations, selected via `--label-mode`:

| Mode | CLI flag | dtype | Shape | Notes |
|------|----------|-------|-------|-------|
| **SOFT** | `--label-mode soft` | float32 | `(M²×12,)` | IDW-weighted distribution over the 3×3 neighbourhood of the true cell. Loss: CrossEntropy. |
| **HARD** | `--label-mode hard` | int64 | scalar | Cardinal class index of the exact cell. Loss: CrossEntropy. |
| **REGRESSION** | `--label-mode regression` | float32 | `(3,)` = `[lat, lon, bearing]` | Raw GPS coordinates and bearing. Loss: MSE on encoded representation. |

---

## Regression Encodings

When using `--label-mode regression`, the `--reg-encoding` flag controls how coordinates are encoded before computing the MSE loss:

| Encoding | CLI flag | Output dim | Description |
|----------|----------|------------|-------------|
| **SCALED** | `--reg-encoding scaled` | 3 | `(lat, lon, bearing)` linearly normalised to `[0, 1]` via geographic bounding box and `bearing / 360`. MLP uses sigmoid output. |
| **SPHERE3D** | `--reg-encoding sphere3d` | 4 | `(lat, lon)` mapped to a unit vector on S² via `latlon_to_unitvec`, bearing normalised to `[0, 1]`. MLP output: L2-normalised xyz + sigmoid bearing. |
| **HYPERSPHERE** | `--reg-encoding hypersphere` | 4 | `(lat, lon, bearing)` mapped onto the unit 3-sphere S³ via `angles_to_hypersphere`. MLP output: L2-normalised 4-dim vector. |

---

## MLP Heads (`Network.py`)

All MLP variants share a 3-layer hidden trunk with **SiLU** activations (`_MLPTrunk`). The head layer and output activation differ by task:

| Class | Label mode | Output dim | Output activation |
|-------|-----------|------------|-------------------|
| `MLPScaled` | REGRESSION / SCALED | 3 | Sigmoid → `[0, 1]³` |
| `MLPSphere3D` | REGRESSION / SPHERE3D | 4 | L2-norm on first 3 dims + sigmoid on last |
| `MLPHypersphere` | REGRESSION / HYPERSPHERE | 4 | L2-norm (unit vector on S³) |
| `MLPClassifier` | SOFT / HARD | M²×12 | Raw logits (no softmax) |

---

## Backbones (`Pretrained_net.py`, `Megaloc.py`)

| Backbone | Factory fn | Architecture | Descriptor dim | Frozen? |
|----------|------------|--------------|---------------|---------|
| **CosPlace** | `CosPlace()` | ResNet-101 | 2048 | Fully frozen |
| **MegaLoc** | `MegaLoc()` | DINOv2-B/14 + SALAD aggregation | 8448 | Fully frozen |

---

## Dataset (`Get_dataset.py`)

`GeoDataset` recursively scans directories for `.jpg` images whose filenames encode GPS and bearing in the `@`-separated format:

```
..@..@..@..@..@<lat>@<lon>@..@..@<bearing>@...jpg
```

Images outside the configured geographic bounding box are automatically discarded. A folder-level lat filter skips entire directories out of range.

### Transforms

| Split | Transform |
|-------|-----------|
| Training (no augmentation) | Resize 252×252 → ToTensor → ImageNet normalise |
| Training (with `--augment`) | Resize 252×252 → RandomChoice(ColorJitter \| GaussianBlur) → ToTensor → ImageNet normalise |
| Validation / Test | Resize 252×252 → ToTensor → ImageNet normalise |

---

## Training (`Training.py`)

- **Optimiser**: Adam, lr = 1e-4, weight_decay = 1e-5
- **Scheduler**: ReduceLROnPlateau (factor = 0.1, patience = 5)
- **Early stopping**: patience = 10 epochs without validation improvement
- **Checkpointing**: best MLP weights saved to `mlp.pth` and synced to wandb
- **Backbone**: kept in `eval()` mode throughout; only the MLP head (and optionally `aggregator.linear` for MegaLoc) is trained
- **Metrics** (wandb): `train_loss`, `val_loss`, `lr`, `epoch_time_min`, and final `recall_at_25m` / `mean_error_m` (regression) or `accuracy` (classification)

---

## Evaluation & Metrics (`Eval.py`, `Utils.py`)

- **Classification**: top-1 accuracy (predicted class == target class)
- **Regression**: Haversine distance between predicted and true `(lat, lon)`. A prediction is considered **correct** if within **25 m** of the ground truth → reported as `Recall@25m`

`visualize_predictions` logs a wandb Table with decoded predictions, ground-truth coordinates, per-sample error, and image thumbnails.

---

## Usage

### Installation

```bash
pip install torch torchvision wandb pillow numpy
```

### Running

```bash
# Regression with scaled encoding, CosPlace backbone
python Main.py --label-mode regression --reg-encoding scaled --backbone cosplace

# Regression with hypersphere encoding, MegaLoc backbone, with augmentation
python Main.py --label-mode regression --reg-encoding hypersphere --backbone megaloc --augment

# Soft-label classification, CosPlace
python Main.py --label-mode soft --backbone cosplace

# Hard-label classification, skip test set evaluation
python Main.py --label-mode hard --backbone cosplace --no-test
```

### CLI Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--backbone` | `cosplace` \| `megaloc` | `cosplace` | VPR backbone |
| `--label-mode` | `soft` \| `hard` \| `regression` | `regression` | Ground-truth label format |
| `--reg-encoding` | `scaled` \| `sphere3d` \| `hypersphere` | `scaled` | Coordinate encoding (regression only) |
| `--augment` | flag | off | Enable colour jitter / Gaussian blur augmentation |
| `--no-test` | flag | off | Skip test-set evaluation |

---

## Project Structure

```
INR_in_VPR/
├── Constants.py       # Bounding box, M, LabelMode, RegressionEncoding enums
├── Get_dataset.py     # GeoDataset, label builders, image transforms
├── Network.py         # MLP heads (Scaled, Sphere3D, Hypersphere, Classifier)
├── VPR_Regressor.py   # Full model: backbone + MLP head
├── Pretrained_net.py  # CosPlace() and MegaLoc() factory functions
├── Megaloc.py         # MegaLoc architecture (DINOv2 + SALAD)
├── Training.py        # train_loop with early stopping and wandb logging
├── Eval.py            # evaluation_loop (validation loss)
├── Utils.py           # Coordinate encoders/decoders, haversine, visualize_predictions
└── Main.py            # Entry point, CLI argument parsing
```

---

## Configuration (`Constants.py`)

| Constant | Value | Meaning |
|----------|-------|---------|
| `LAT_MIN` / `LAT_MAX` | 37.7575 / 37.7625 | Latitude bounding box |
| `LON_MIN` / `LON_MAX` | -122.4325 / -122.4275 | Longitude bounding box |
| `M` | Grid resolution (M × M spatial cells) |
