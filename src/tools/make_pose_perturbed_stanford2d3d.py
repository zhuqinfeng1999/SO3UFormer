import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser("Make pose-perturbed Stanford2D3D dataset (ERP remap)")
    parser.add_argument("--dataset_root_dir", type=str, default="./data")
    parser.add_argument("--src_dataset_name", type=str, default="stanford2d3d")
    parser.add_argument("--dst_dataset_name", type=str, default="stanford2d3d_pose20_axisU_seed0")
    parser.add_argument("--max_deg", type=float, default=20.0)
    parser.add_argument("--mode", type=str, default="axis_angle_uniform",
                        choices=["axis_angle_uniform", "axis_angle_trunc_gauss", "euler_uniform"])
    parser.add_argument("--sigma_deg", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--splits", type=str, default="train,val")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--overwrite", type=int, default=0)
    parser.add_argument("--dry_run", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0, help="limit number of samples (0 = all)")
    return parser.parse_args()


def read_list(list_file: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(list_file, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def resolve_list_file(dataset_root_dir: str, dataset_name: str, split: str) -> str:
    candidate = os.path.join(dataset_root_dir, dataset_name, "splits_2d3d", f"stanford2d3d_{split}.txt")
    if os.path.isfile(candidate):
        return candidate
    return os.path.join(dataset_root_dir, "splits_2d3d", f"stanford2d3d_{split}.txt")


def hash_to_uint32(text: str) -> int:
    h = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False)


def axis_angle_to_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)


def euler_zyx_to_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    rx = math.radians(roll_deg)
    ry = math.radians(pitch_deg)
    rz = math.radians(yaw_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    return Rz @ Ry @ Rx


def rotmat_to_euler_zyx(R: np.ndarray) -> Tuple[float, float, float]:
    # returns roll, pitch, yaw in degrees (x, y, z)
    if abs(R[2, 0]) < 1.0 - 1e-8:
        pitch = -math.asin(R[2, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = -math.asin(R[2, 0])
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def sample_rotation(rel_path: str, seed: int, mode: str, max_deg: float, sigma_deg: float) -> Dict:
    per_seed = (seed + hash_to_uint32(rel_path)) % (2 ** 32)
    rng = np.random.RandomState(per_seed)

    if mode == "euler_uniform":
        roll = rng.uniform(-max_deg, max_deg)
        pitch = rng.uniform(-max_deg, max_deg)
        yaw = rng.uniform(-max_deg, max_deg)
        R = euler_zyx_to_matrix(roll, pitch, yaw)
        axis = None
        angle_deg = None
        euler_deg = [roll, pitch, yaw]
    else:
        # axis-angle variants
        axis = rng.normal(size=3)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        if mode == "axis_angle_trunc_gauss":
            angle = None
            while angle is None or angle > max_deg:
                angle = abs(rng.normal(loc=0.0, scale=sigma_deg))
            angle_deg = angle
        else:
            angle_deg = rng.uniform(0.0, max_deg)
        angle_rad = math.radians(angle_deg)
        R = axis_angle_to_matrix(axis, angle_rad)
        roll, pitch, yaw = rotmat_to_euler_zyx(R)
        euler_deg = [roll, pitch, yaw]

    return {
        "R": R,
        "axis": axis,
        "angle_deg": angle_deg,
        "euler_deg": euler_deg,
        "seed": per_seed,
    }


_dir_cache_np: Dict[Tuple[int, int], np.ndarray] = {}


def get_dirs_np(h: int, w: int) -> np.ndarray:
    key = (h, w)
    if key in _dir_cache_np:
        return _dir_cache_np[key]
    u = (np.arange(w, dtype=np.float64) + 0.5) / w
    v = (np.arange(h, dtype=np.float64) + 0.5) / h
    theta = (u * 2.0 * math.pi) - math.pi
    phi = v * math.pi
    theta, phi = np.meshgrid(theta, phi)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    dirs = np.stack([x, y, z], axis=-1)
    _dir_cache_np[key] = dirs
    return dirs


def build_remap_np(h: int, w: int, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dirs = get_dirs_np(h, w)
    R_inv = R.T
    d_in = dirs @ R_inv.T
    x, y, z = d_in[..., 0], d_in[..., 1], d_in[..., 2]
    theta = np.arctan2(y, x)
    z = np.clip(z, -1.0, 1.0)
    phi = np.arccos(z)

    u = (theta + math.pi) / (2.0 * math.pi) * w - 0.5
    v = (phi / math.pi) * h - 0.5
    u = np.mod(u, w)
    v = np.clip(v, 0.0, h - 1.0)
    return u.astype(np.float32), v.astype(np.float32)


_dir_cache_torch: Dict[Tuple[int, int, str], torch.Tensor] = {}


def get_dirs_torch(h: int, w: int, device: torch.device) -> torch.Tensor:
    key = (h, w, str(device))
    if key in _dir_cache_torch:
        return _dir_cache_torch[key]
    u = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    v = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    theta = (u * 2.0 * math.pi) - math.pi
    phi = v * math.pi
    theta, phi = torch.meshgrid(theta, phi, indexing="xy")
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    dirs = torch.stack([x, y, z], dim=-1)
    _dir_cache_torch[key] = dirs
    return dirs


def build_grid_torch(h: int, w: int, R: np.ndarray, device: torch.device) -> torch.Tensor:
    dirs = get_dirs_torch(h, w, device)
    R_inv = torch.tensor(R.T, device=device, dtype=torch.float32)
    d_in = torch.einsum("hwc,dc->hwd", dirs, R_inv)
    x, y, z = d_in[..., 0], d_in[..., 1], d_in[..., 2]
    theta = torch.atan2(y, x)
    z = torch.clamp(z, -1.0, 1.0)
    phi = torch.acos(z)

    u = (theta + math.pi) / (2.0 * math.pi) * w - 0.5
    v = (phi / math.pi) * h - 0.5
    u = torch.remainder(u, w)
    v = torch.clamp(v, 0.0, h - 1.0)

    grid_x = (u / (w - 1)) * 2.0 - 1.0
    grid_y = (v / (h - 1)) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    return grid


def remap_cv2(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, interp: int) -> np.ndarray:
    return cv2.remap(img, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def remap_torch(img: np.ndarray, grid: torch.Tensor, mode: str, device: torch.device) -> np.ndarray:
    if img.ndim == 2:
        img_t = torch.from_numpy(img).to(device=device, dtype=torch.float32)[None, None]
    else:
        img_t = torch.from_numpy(img).to(device=device, dtype=torch.float32).permute(2, 0, 1)[None]
    out = F.grid_sample(img_t, grid, mode=mode, align_corners=True)
    out = out.squeeze(0)
    if out.ndim == 2:
        result = out.cpu().numpy()
    else:
        result = out.permute(1, 2, 0).cpu().numpy()
    return result


def process_one(task: Dict) -> Dict:
    rel_rgb = task["rel_rgb"]
    rel_depth = task["rel_depth"]
    rel_sem = task["rel_sem"]
    src_root = task["src_root"]
    dst_root = task["dst_root"]
    mode = task["mode"]
    max_deg = task["max_deg"]
    sigma_deg = task["sigma_deg"]
    seed = task["seed"]
    overwrite = task["overwrite"]
    use_torch = task["use_torch"]
    device = task["device"]

    src_rgb = os.path.join(src_root, rel_rgb)
    src_depth = os.path.join(src_root, rel_depth)
    src_sem = os.path.join(src_root, rel_sem)

    dst_rgb = os.path.join(dst_root, rel_rgb)
    dst_depth = os.path.join(dst_root, rel_depth)
    dst_sem = os.path.join(dst_root, rel_sem)

    os.makedirs(os.path.dirname(dst_rgb), exist_ok=True)
    os.makedirs(os.path.dirname(dst_depth), exist_ok=True)
    os.makedirs(os.path.dirname(dst_sem), exist_ok=True)

    already = os.path.isfile(dst_rgb) and os.path.isfile(dst_depth) and os.path.isfile(dst_sem)
    if already and not overwrite:
        meta = sample_rotation(rel_rgb, seed, mode, max_deg, sigma_deg)
        return {
            "rel_rgb_path": rel_rgb,
            "rel_depth_path": rel_depth,
            "rel_label_path": rel_sem,
            "mode": mode,
            "max_deg": max_deg,
            "seed": seed,
            "R": meta["R"].reshape(-1).tolist(),
            "axis": None if meta["axis"] is None else meta["axis"].tolist(),
            "angle_deg": meta["angle_deg"],
            "euler_deg": meta["euler_deg"],
            "skipped": True,
        }

    rgb = cv2.imread(src_rgb, cv2.IMREAD_COLOR)
    depth = cv2.imread(src_depth, cv2.IMREAD_UNCHANGED)
    sem = cv2.imread(src_sem, cv2.IMREAD_COLOR)

    if rgb is None or depth is None or sem is None:
        raise RuntimeError(f"Failed reading {rel_rgb}")

    h, w = rgb.shape[:2]
    meta = sample_rotation(rel_rgb, seed, mode, max_deg, sigma_deg)
    R = meta["R"]

    if use_torch:
        grid = build_grid_torch(h, w, R, device)
        rgb_out = remap_torch(rgb, grid, mode="bilinear", device=device)
        sem_out = remap_torch(sem, grid, mode="nearest", device=device)
        depth_out = remap_torch(depth, grid, mode="bilinear", device=device)
    else:
        map_x, map_y = build_remap_np(h, w, R)
        rgb_out = remap_cv2(rgb, map_x, map_y, interp=cv2.INTER_LINEAR)
        sem_out = remap_cv2(sem, map_x, map_y, interp=cv2.INTER_NEAREST)
        depth_out = remap_cv2(depth, map_x, map_y, interp=cv2.INTER_LINEAR)

    # restore dtypes
    rgb_out = np.clip(rgb_out, 0, 255).astype(np.uint8)
    sem_out = np.clip(sem_out, 0, 255).astype(np.uint8)
    if depth.dtype == np.uint16:
        depth_out = np.rint(depth_out).astype(np.uint16)
    else:
        depth_out = depth_out.astype(depth.dtype)

    cv2.imwrite(dst_rgb, rgb_out)
    cv2.imwrite(dst_depth, depth_out)
    cv2.imwrite(dst_sem, sem_out)

    return {
        "rel_rgb_path": rel_rgb,
        "rel_depth_path": rel_depth,
        "rel_label_path": rel_sem,
        "mode": mode,
        "max_deg": max_deg,
        "seed": seed,
        "R": R.reshape(-1).tolist(),
        "axis": None if meta["axis"] is None else meta["axis"].tolist(),
        "angle_deg": meta["angle_deg"],
        "euler_deg": meta["euler_deg"],
        "skipped": False,
    }


def main():
    args = parse_args()

    src_root = os.path.join(args.dataset_root_dir, args.src_dataset_name)
    dst_root = os.path.join(args.dataset_root_dir, args.dst_dataset_name)

    if args.src_dataset_name == args.dst_dataset_name:
        print("[ERR] src_dataset_name equals dst_dataset_name")
        sys.exit(1)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    list_files = {split: resolve_list_file(args.dataset_root_dir, args.src_dataset_name, split) for split in splits}

    for split, lf in list_files.items():
        if not os.path.isfile(lf):
            print(f"[ERR] missing list file for split {split}: {lf}")
            sys.exit(1)

    samples = {}
    for split, lf in list_files.items():
        for rgb_rel, depth_rel in read_list(lf):
            samples[rgb_rel] = depth_rel

    tasks = []
    for rgb_rel, depth_rel in samples.items():
        sem_rel = depth_rel.replace("depth", "semantic")
        tasks.append({
            "rel_rgb": rgb_rel,
            "rel_depth": depth_rel,
            "rel_sem": sem_rel,
            "src_root": src_root,
            "dst_root": dst_root,
            "mode": args.mode,
            "max_deg": args.max_deg,
            "sigma_deg": args.sigma_deg,
            "seed": args.seed,
            "overwrite": bool(args.overwrite),
            "use_torch": args.device.startswith("cuda") and torch.cuda.is_available(),
            "device": torch.device(args.device if torch.cuda.is_available() else "cpu"),
        })

    if args.max_samples and args.max_samples > 0:
        tasks = tasks[:args.max_samples]

    print(f"[INFO] total samples: {len(tasks)}")
    print("[INFO] first 5 samples:")
    for t in tasks[:5]:
        print("  ", t["rel_rgb"])

    if not tasks:
        print("[WARN] no samples to process")
        return

    if args.dry_run:
        print("[DRY_RUN] no files written")
        return

    os.makedirs(dst_root, exist_ok=True)
    split_dst_dir = os.path.join(dst_root, "splits_2d3d")
    os.makedirs(split_dst_dir, exist_ok=True)
    for split, lf in list_files.items():
        dst_lf = os.path.join(split_dst_dir, os.path.basename(lf))
        shutil.copyfile(lf, dst_lf)

    meta_path = os.path.join(dst_root, "pose_perturb_meta.jsonl")

    use_torch = tasks[0]["use_torch"] if tasks else False
    if use_torch and args.num_workers > 0:
        print("[WARN] CUDA device selected; forcing num_workers=0 to avoid multi-process GPU contention")
        args.num_workers = 0

    processed = 0
    with open(meta_path, "w") as meta_f:
        if args.num_workers > 0:
            import multiprocessing as mp
            with mp.Pool(args.num_workers) as pool:
                for meta in pool.imap_unordered(process_one, tasks, chunksize=4):
                    meta_f.write(json.dumps(meta) + "\n")
                    processed += 1
                    if processed % 200 == 0:
                        print(f"[INFO] processed {processed}/{len(tasks)}")
        else:
            for t in tasks:
                try:
                    meta = process_one(t)
                except Exception as e:
                    print(f"[ERR] {t['rel_rgb']}: {e}")
                    continue
                meta_f.write(json.dumps(meta) + "\n")
                processed += 1
                if processed % 200 == 0:
                    print(f"[INFO] processed {processed}/{len(tasks)}")

    print(f"[DONE] wrote pose-perturbed dataset to {dst_root}")
    print(f"[DONE] metadata: {meta_path}")


if __name__ == "__main__":
    main()
