# SO3UFormer (Pose35, main model)

This release packages the **SO3UFormer** code, the **Pose35** data construction script, the **SO(3) full3d** evaluation script, and a **pretrained checkpoint**.

## Contents
- `src/` core code (training, model, metrics)
- `src/tools/`
  - `make_pose_perturbed_stanford2d3d.py` — Pose35 dataset generation
  - `rotation_sensitivity.py` — SO(3) full3d evaluation
- `scripts/` convenience scripts
  - `make_pose35.sh` — generate Pose35 from Stanford2D3D
  - `train_so3uformer.sh` — train the main model
  - `eval_so3_full3d.sh` — run SO(3) full3d evaluation
- `pretrained/model_best_miou.pth` — best checkpoint for the main model
- `logs/training_log.txt` — sanitized training log

## Requirements
- Python 3.8+
- PyTorch + torchvision (install matching your CUDA)

Install remaining dependencies:

```bash
pip install -r requirements.txt
```

## Dataset: Stanford2D3D
Download **Stanford2D3D** and place it under your data root:

```
<data_root>/stanford2d3d/
```


## Pose35 Dataset Generation
This project uses a pose‑perturbed dataset named:

```
stanford2d3d_pose35_axisU_seed0
```

Generate it with:

```bash
DATA_ROOT=<data_root> \
  bash scripts/make_pose35.sh
```

This creates:

```
<data_root>/stanford2d3d_pose35_axisU_seed0
```

## Training (Main Model)
The main model configuration uses:
- **No absolute latitude PE** (`phi0`)
- **Gauge‑pooled Fourier bias** (`gauge_pool`)
- **Quadrature attention (logit mode)**
- **Geometry‑consistent sampling** (area downsample + geodesic kernel upsample)
- **SO(3) consistency loss** weight = 0.05

Train with:

```bash
DATA_ROOT=<data_root> \
OUT_DIR=./outputs/so3uformer \
NPROC_PER_NODE=2 \
MASTER_PORT=29621 \
  bash scripts/train_so3uformer.sh
```

## SO(3) Full3D Evaluation
Run the SO(3) stress test (full3d group) with:

```bash
DATA_ROOT=<data_root> \
WEIGHTS=./pretrained/model_best_miou.pth \
OUT_DIR=./outputs/so3_eval \
  bash scripts/eval_so3_full3d.sh
```

The script writes a JSON summary and plots under `OUT_DIR`.

## Pretrained & Logs
- **Checkpoint:** `pretrained/model_best_miou.pth`
- **Training log:** `logs/training_log.txt`
