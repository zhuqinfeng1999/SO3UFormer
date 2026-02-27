#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./data}"
WEIGHTS="${WEIGHTS:-./pretrained/model_best_miou.pth}"
OUT_DIR="${OUT_DIR:-./outputs/so3_eval}"

python3 src/tools/rotation_sensitivity.py \
  --dataset_root_dir "${DATA_ROOT}" \
  --dataset_name stanford2d3d_pose35_axisU_seed0 \
  --weights_path "${WEIGHTS}" \
  --log_dir "${OUT_DIR}" \
  --eval_split val \
  --batch_size 8 \
  --num_workers 8 \
  --repeats 3 \
  --full3d_samples 10 \
  --seed 123 \
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
  --upsample_sigma 0.4
