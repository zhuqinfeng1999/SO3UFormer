
<a id="english"></a>

<p align="center">
  <a href="#english"><b>English</b></a> ｜ <a href="#中文"><b>中文</b></a>
</p>

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

## 🔥 Key Ideas

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


---

<a id="中文"></a>

<p align="center">
  <a href="#english"><b>English</b></a> ｜ <a href="#中文"><b>中文</b></a>
</p>

# SO3UFormer：抗旋转全景分割

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

**SO3UFormer 与 Pose35 协议的官方代码发布**  
Pose35 生成 • 训练 • **OOD SO(3) 压力测试**评估 • 预训练权重

**论文：** *SO3UFormer: Learning Intrinsic Spherical Features for Rotation-Robust Panoramic Segmentation*  
**arXiv：** https://arxiv.org/abs/2602.22867

</div>

---

## ✨ 什么是 SO3UFormer？

全景分割模型往往在训练时隐含了 *直立 / 与重力对齐* 的假设。  
当相机发生滚转–俯仰变化（例如无人机倾斜飞行、手持抖动）时，许多方法会将语义泄露到某个特权参考系中，从而在完整的 3D 旋转重定向下失效。

SO3UFormer 旨在学习**对旋转鲁棒的球面特征**，并在 **Pose35**（Stanford2D3D 的姿态扰动变体）上通过 **分布外（OOD）SO(3) 压力测试**来评估鲁棒性。

---

## 🔥 核心思想

SO3UFormer 的主模型由五个要素组成：

- **不使用绝对纬度位置编码**（移除最强的“重力线索”捷径）
- **与求积一致的局部注意力**（使用均值归一化的网格面积权重进行 logit 校正）
- **规约（gauge）池化的傅里叶相对偏置**（切平面投影角度 + 离散规约池化；不依赖全局坐标轴）
- **几何一致的多尺度采样**（面积加权下采样 + 测地核上采样）
- **训练时 SO(3) 一致性正则项**（基于索引的球面重采样下，在 logit 空间做 MSE）

> [!NOTE]
> 本仓库聚焦于一个*实用的鲁棒性设置*：在 Pose35（±35° 姿态扰动）上训练，并使用**完整 SO(3)** 的 OOD 压力测试进行评估。

---

## 🖼️ 图示

<p align="center">
  <img src="figures/fig2.jpg" width="92%" />
</p>
<p align="center"><i>SO3UFormer 总览：规约感知注意力、几何一致采样与 SO(3) 一致性正则。</i></p>

<p align="center">
  <img src="figures/fig3.jpg" width="92%" />
</p>
<p align="center"><i>Pose35 验证集全景图上的 SO(3) 压力测试定性对比。</i></p>

---

## 🧭 目录

- [🧩 预训练模型](#-预训练模型)
- [⚡ 快速开始](#-快速开始)
- [📦 安装](#-安装)
- [🗂️ 数据集准备](#️-数据集准备)
- [🧪 Pose35 生成](#-pose35-生成)
- [▶️ 训练（主模型）](#-训练主模型)
- [🧪 评估：OOD SO(3) 压力测试（Full3D）](#-评估-ood-so3-压力测试-full3d)
- [📊 结果](#-结果)
- [📁 仓库结构](#-仓库结构)
- [🧾 引用](#-引用)
- [📬 联系方式](#-联系方式)
- [🙏 致谢](#-致谢)
- [📄 许可证](#-许可证)

---
## 🧩 预训练模型

下载预训练权重（Google Drive），并放置到：

- `pretrained/model_best_miou.pth`

**下载：** https://drive.google.com/file/d/1wY-MtVnu41SJbzp8o0sCNuEvXuwgKvWJ/view?usp=sharing

> [!NOTE]
> 评估脚本默认使用 `WEIGHTS=./pretrained/model_best_miou.pth`。

---

## ⚡ 快速开始

> [!TIP]
> 如果你已经（i）将 Stanford2D3D 下载到 `DATA_ROOT` 下，（ii）完成 Pose35 生成，（iii）把预训练权重放在 `./pretrained/model_best_miou.pth`，那么可以一条命令运行 OOD SO(3) 压力测试。

```bash
DATA_ROOT=<data_root> \
WEIGHTS=./pretrained/model_best_miou.pth \
OUT_DIR=./outputs/so3_eval \
  bash scripts/eval_so3_full3d.sh
````

---

## 📦 安装

### 环境需求

* Python **3.8+**
* PyTorch + torchvision（需匹配你的 CUDA 运行时）

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 🗂️ 数据集准备

下载 **Stanford2D3D**，并放到你的数据根目录下：

```text
<data_root>/
  stanford2d3d/
    ...
```

Pose35 会在同一个 `<data_root>` 下与其并列生成。

> [!IMPORTANT]
> 我们遵循 Stanford2D3D 的语义设置，共 **13 类**；其中 **unknown** 标签作为 **0** 类并在 mIoU 中忽略。

---

## 🧪 Pose35 生成

使用以下命令生成 Pose35：

```bash
DATA_ROOT=<data_root> \
  bash scripts/make_pose35.sh
```

将生成：

```text
<data_root>/stanford2d3d_pose35_axisU_seed0
```

---

## ▶️ 训练（主模型）

主模型配置包含：

* **不使用绝对纬度位置编码**（`phi0`）
* **规约池化傅里叶偏置**（`gauge_pool`）
* **求积注意力（logit 模式）**
* **几何一致采样**（面积下采样 + 测地核上采样）
* **SO(3) 一致性损失**权重 = 0.05

训练命令：

```bash
DATA_ROOT=<data_root> \
OUT_DIR=./outputs/so3uformer \
NPROC_PER_NODE=2 \
MASTER_PORT=29621 \
  bash scripts/train_so3uformer.sh
```

---

## 🧪 评估：OOD SO(3) 压力测试（Full3D）

运行 SO(3) 压力测试（full3d group）：

```bash
DATA_ROOT=<data_root> \
WEIGHTS=./pretrained/model_best_miou.pth \
OUT_DIR=./outputs/so3_eval \
  bash scripts/eval_so3_full3d.sh
```
---

## 📊 结果

### 主消融实验（Pose35）


> 说明：最后一行启用了 SO(3) 一致性正则项（\mathcal{L}_{eq}），并使用固定权重（\lambda=0.05）。

| 无绝对纬度 PE | 求积注意力 | 规约池化偏置 | 几何采样 | 𝓛<sub>eq</sub> | Base mIoU | SO(3) mIoU |
| ------------: | ---------: | -----------: | -------: | --------------: | --------: | ---------: |
|               |            |              |          |                 |     67.53 |      25.26 |
|             ✅ |            |              |          |                 |     68.64 |      64.66 |
|             ✅ |          ✅ |              |          |                 |     70.05 |      65.20 |
|             ✅ |          ✅ |            ✅ |          |                 |     70.42 |      69.72 |
|             ✅ |          ✅ |            ✅ |       ✅ |                 |     70.92 |      69.90 |
|             ✅ |          ✅ |            ✅ |       ✅ |               ✅ | **72.03** |  **70.67** |

---

### 对比实验（均在 Pose35 上重新训练）

| 方法                     |      发表/状态 | Base mIoU | SO(3) mIoU |
| ------------------------ | -------------: | --------: | ---------: |
| SFSS                     |        WACV 2024 |     42.02 |      30.99 |
| HealSwin                 |        CVPR 2024 |     62.45 |      30.55 |
| Elite360                 |        CVPR 2024 |     67.39 |      25.71 |
| SphereUFormer (baseline) |        CVPR 2025 |     67.53 |      25.26 |
| **SO3UFormer (ours)**    | **Under review** | **72.03** |  **70.67** |

---

## 📁 仓库结构

```text
.
├── src/                       # 核心训练与模型代码
│   ├── network/               # 模型（SphereUFormer/SO3UFormer）
│   └── zqf_tools/             # Pose35 生成 + SO(3) 压力测试
├── pretrained/                # 预训练权重（放在这里）
├── figures/                   # 论文图（fig1/fig2/fig3）
├── requirements.txt
└── README.md
```

---

## 🧾 引用

如果你觉得该项目有帮助，请引用：

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

## 📬 联系方式

* Qinfeng Zhu
* Yunxi Jiang
* Lei Fan（通讯作者）

（有关环境配置、复现或评估的问题，请优先在仓库里开 issue。）

---

## 🙏 致谢

感谢 **SphereUFormer** 作者开源了基线架构实现，我们将其作为球面 U 形骨干网络与基线对比的参考代码库：
[https://github.com/yanivbenny/sphere_uformer](https://github.com/yanivbenny/sphere_uformer)

---

## 📄 许可证

参见 `LICENSE`。
`````
