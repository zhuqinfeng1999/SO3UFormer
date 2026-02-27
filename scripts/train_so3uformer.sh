#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./data}"
OUT_DIR="${OUT_DIR:-./outputs/q21}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29621}"

EXP_NAME="pose35__so3uformer__B2-gaugePoolInv__phi0__A1-qlogit__Cboth__D-w0p05"

TORCHRUN=${TORCHRUN:-torchrun}

${TORCHRUN} --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
  src/train.py --distributed --task segmentation \
  --dataset_root_dir "${DATA_ROOT}" \
  --dataset_name stanford2d3d_pose35_axisU_seed0 \
  --log_dir "${OUT_DIR}" \
  --exp_name "${EXP_NAME}" \
  --use_quadrature_attn 1 \
  --quadrature_mode logit \
  --use_abs_phi_pe 0 \
  --rel_pos_bias_type gauge_pool \
  --rel_pos_bins 32 \
  --gauge_mode pool_invariant \
  --gauge_num_frames 3 \
  --gauge_m_max 2 \
  --gauge_anchor_mode tangent_max \
  --downsample_mode area_avg \
  --upsample_mode geodesic_kernel \
  --upsample_sigma 0.4 \
  --eq_loss_weight 0.05 \
  --eq_loss_samples 1
