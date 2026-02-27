#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./data}"

python3 src/tools/make_pose_perturbed_stanford2d3d.py \
  --dataset_root_dir "${DATA_ROOT}" \
  --src_dataset_name stanford2d3d \
  --dst_dataset_name stanford2d3d_pose35_axisU_seed0 \
  --max_deg 35 \
  --mode axis_angle_uniform \
  --seed 0 \
  --splits train,val
