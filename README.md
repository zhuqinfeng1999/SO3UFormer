
# SO3UFormer: Rotation-Robust Panoramic Segmentation

<p>
  <a href="https://arxiv.org/abs/2602.22867">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2602.22867-b31b1b.svg" />
  </a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C?logo=pytorch&logoColor=white" />
  <img alt="Task" src="https://img.shields.io/badge/Task-Panoramic%20Segmentation-00A3E0" />
  <img alt="Domain" src="https://img.shields.io/badge/Domain-360%2F%20Spherical%20Vision-4CAF50" />
  <img alt="Robustness" src="https://img.shields.io/badge/Robustness-SO(3)%20OOD%20Stress%20Test-1F4E79" />
  <img alt="PRs" src="https://img.shields.io/badge/PRs-Welcome-brightgreen" />
</p>

**Official code release for SO3UFormer and the Pose35 protocol**  
Pose35 generation • Training • **OOD SO(3) stress test** evaluation • Pretrained checkpoint

**Paper:** *SO3UFormer: Learning Intrinsic Spherical Features for Rotation-Robust Panoramic Segmentation*  
**arXiv:** https://arxiv.org/abs/2602.22867

</div>

---

## ✨ What is SO3UFormer?

Panoramic segmentation models are often trained under an implicit *upright / gravity-aligned* assumption.  
When the camera undergoes roll–pitch changes (e.g., drones banking, handheld jitter), many methods leak semantics into a privileged frame and can fail under full 3D reorientations.

SO3UFormer targets **rotation-robust spherical features** and evaluates robustness with an **out-of-distribution (OOD) SO(3) stress test** on **Pose35**, a pose-perturbed variant of Stanford2D3D.

---

## 🔥 Key Ideas (implementation-faithful)

SO3UFormer combines five ingredients (as used in our main model):

- **No absolute latitude positional encoding** (removes the strongest “gravity cue” shortcut)
- **Quadrature-consistent local attention** (logit correction with mean-normalized mesh area weights)
- **Gauge-pooled Fourier relative bias** (tangent-plane projected angles + discrete gauge pooling; no global axes)
- **Geometry-consistent multi-scale sampling** (area-weighted downsample + geodesic-kernel upsample)
- **Training-time SO(3) consistency regularizer** (logit-space MSE under index-based spherical resampling)

> [!NOTE]
> This repo focuses on a *practical robustness setting*: we train on Pose35 (±35° pose perturbations) and evaluate with a **full SO(3)** OOD stress test.

---

## 🖼️ Figures

<p align="center">
  <img src="figures/fig2.jpg" width="92%" />
</p>
<p align="center"><i>SO3UFormer overview: gauge-aware attention, geometry-consistent sampling, and SO(3) consistency regularization.</i></p>

<p align="center">
  <img src="figures/fig3.jpg" width="92%" />
</p>
<p align="center"><i>Qualitative SO(3) stress test comparisons on Pose35 validation panoramas.</i></p>

---

## 🧭 Table of Contents

- [🧩 Pretrained Model](#-pretrained-model)
- [⚡ Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🗂️ Dataset Preparation](#️-dataset-preparation)
- [🧪 Pose35 Generation](#-pose35-generation)
- [▶️ Training (Main Model)](#-training-main-model)
- [🧪 Evaluation: OOD SO(3) Stress Test (Full3D)](#-evaluation-ood-so3-stress-test-full3d)
- [📊 Results](#-results)
- [📁 Repo Structure](#-repo-structure)
- [🧾 Citation](#-citation)
- [📬 Contact](#-contact)
- [🙏 Acknowledgement](#-acknowledgement)
- [📄 License](#-license)

---
## 🧩 Pretrained Model

Download the pretrained checkpoint (Google Drive) and place it at:

- `pretrained/model_best_miou.pth`

**Download:** https://drive.google.com/file/d/1wY-MtVnu41SJbzp8o0sCNuEvXuwgKvWJ/view?usp=sharing

> [!NOTE]
> The evaluation scripts expect `WEIGHTS=./pretrained/model_best_miou.pth`.

---

## ⚡ Quick Start

> [!TIP]
> If you already have (i) Stanford2D3D downloaded under `DATA_ROOT`, (ii) Pose35 generated, and (iii) the pretrained weights placed at `./pretrained/model_best_miou.pth`, you can run the OOD SO(3) stress test in one command.

```bash
DATA_ROOT=<data_root> \
WEIGHTS=./pretrained/model_best_miou.pth \
OUT_DIR=./outputs/so3_eval \
  bash scripts/eval_so3_full3d.sh
````

---

## 📦 Installation

### Requirements

* Python **3.8+**
* PyTorch + torchvision (match your CUDA runtime)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🗂️ Dataset Preparation

Download **Stanford2D3D** and place it under your data root:

```text
<data_root>/
  stanford2d3d/
    ...
```

Pose35 will be created alongside it under the same `<data_root>`.

> [!IMPORTANT]
> We follow the Stanford2D3D semantic setup with **13 classes**; the **unknown** label is treated as class **0** and ignored in mIoU.

---

## 🧪 Pose35 Generation

Generate Pose35 with:

```bash
DATA_ROOT=<data_root> \
  bash scripts/make_pose35.sh
```

This creates:

```text
<data_root>/stanford2d3d_pose35_axisU_seed0
```

---

## ▶️ Training (Main Model)

The main model configuration uses:

* **No absolute latitude PE** (`phi0`)
* **Gauge-pooled Fourier bias** (`gauge_pool`)
* **Quadrature attention (logit mode)**
* **Geometry-consistent sampling** (area downsample + geodesic kernel upsample)
* **SO(3) consistency loss** weight = 0.05

Train with:

```bash
DATA_ROOT=<data_root> \
OUT_DIR=./outputs/so3uformer \
NPROC_PER_NODE=2 \
MASTER_PORT=29621 \
  bash scripts/train_so3uformer.sh
```

---

## 🧪 Evaluation: OOD SO(3) Stress Test (Full3D)

Run the SO(3) stress test (full3d group) with:

```bash
DATA_ROOT=<data_root> \
WEIGHTS=./pretrained/model_best_miou.pth \
OUT_DIR=./outputs/so3_eval \
  bash scripts/eval_so3_full3d.sh
```
---

## 📊 Results

### Main ablation (Pose35)


> Notes: the last row enables the SO(3)-consistency regularizer (\mathcal{L}_{eq}) with a fixed weight (\lambda=0.05).

| No abs. lat. PE | Quadrature attn. | Gauge-pooled bias | Geo. sampling | 𝓛<sub>eq</sub> | Base mIoU | SO(3) mIoU |
| --------------: | ---------------: | ----------------: | ------------: | --------------: | --------: | ---------: |
|                 |                  |                   |               |                 |     67.53 |      25.26 |
|               ✅ |                  |                   |               |                 |     68.64 |      64.66 |
|               ✅ |                ✅ |                   |               |                 |     70.05 |      65.20 |
|               ✅ |                ✅ |                 ✅ |               |                 |     70.42 |      69.72 |
|               ✅ |                ✅ |                 ✅ |             ✅ |                 |     70.92 |      69.90 |
|               ✅ |                ✅ |                 ✅ |             ✅ |               ✅ | **72.03** |  **70.67** |

---

### Comparison (all retrained on Pose35)

| Method                   |      Publication | Base mIoU | SO(3) mIoU |
| ------------------------ | ---------------: | --------: | ---------: |
| SFSS                     |        WACV 2024 |     42.02 |      30.99 |
| HealSwin                 |        CVPR 2024 |     62.45 |      30.55 |
| Elite360                 |        CVPR 2024 |     67.39 |      25.71 |
| SphereUFormer (baseline) |        CVPR 2025 |     67.53 |      25.26 |
| **SO3UFormer (ours)**    | **Under review** | **72.03** |  **70.67** |

---

## 📁 Repo Structure

```text
.
├── src/                       # core training + model code
│   ├── network/               # models (SphereUFormer/SO3UFormer)
│   └── zqf_tools/             # Pose35 generation + SO(3) stress test
├── pretrained/                # pretrained weights (place here)
├── figures/                   # paper figures (fig1/fig2/fig3)
├── requirements.txt
└── README.md
```

---

## 🧾 Citation

If you find this project useful, please cite:

```bibtex
@article{zhu2026so3uformer,
  title   = {SO3UFormer: Learning Intrinsic Spherical Features for Rotation-Robust Panoramic Segmentation},
  author  = {Zhu, Qinfeng and Jiang, Yunxi and Fan, Lei},
  journal = {arXiv preprint arXiv:2602.22867},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.22867}
}
```

---

## 📬 Contact

* Qinfeng Zhu
* Yunxi Jiang
* Lei Fan (corresponding)

(Please open an issue for questions about setup, reproduction, or evaluation.)

---

## 🙏 Acknowledgement

We thank the authors of **SphereUFormer** for releasing the baseline architecture implementation, which we used as a reference codebase for the spherical U-shaped backbone and baseline comparisons:
[https://github.com/yanivbenny/sphere_uformer](https://github.com/yanivbenny/sphere_uformer)

---

## 📄 License

See `LICENSE`.

```
```
