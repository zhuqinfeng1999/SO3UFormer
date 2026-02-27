import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional

# ensure repo src is on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_SRC = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_SRC not in sys.path:
    sys.path.append(PROJECT_SRC)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation
from scipy import stats
import yaml
import shlex
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from data.stanford2d3d import Stanford2D3D
from metrics.segmentation import ConfusionMatrix, compute_segmentation_metrics
from network.sphere_model import SO3UFormer
from trimesh_utils import IcoSphereRef


def parse_args():
    parser = argparse.ArgumentParser(description="Rotation sensitivity experiments on Stanford2D3D val set")
    parser.add_argument("--dataset_root_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="stanford2d3d")
    parser.add_argument("--weights_path", type=str, required=True, help="path to trained model .pth")
    parser.add_argument("--log_dir", type=str, default="../outputs/exp", help="where to save logs/plots")
    parser.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--img_rank", type=int, default=None, help="model sphere rank used in training (auto if not set)")
    parser.add_argument("--eval_ranks", type=str, default="7,8", help="comma separated ranks for multi-resolution group")
    parser.add_argument("--node_type", type=str, default=None, choices=["face", "vertex"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--repeats", type=int, default=3, help="runs per group")
    parser.add_argument("--full3d_samples", type=int, default=10)
    parser.add_argument("--noise_std", type=float, default=0.05, help="gaussian noise std for noise group (on normalized rgb)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--strict_load", type=int, default=1, help="strict state_dict load")
    parser.add_argument("--baseline_only", type=int, default=0, help="evaluate only baseline (identity rotation)")

    # ablation flags (auto by default)
    parser.add_argument("--use_quadrature_attn", type=int, default=None)
    parser.add_argument("--quadrature_mode", type=str, default=None, choices=["logit", "value", "value_renorm"])
    parser.add_argument("--use_abs_phi_pe", type=int, default=None)
    parser.add_argument("--rel_pos_bias_type", type=str, default=None)
    parser.add_argument("--rel_pos_bins", type=int, default=None)
    parser.add_argument("--gauge_num_frames", type=int, default=None)
    parser.add_argument("--gauge_m_max", type=int, default=None)
    parser.add_argument("--gauge_mode", type=str, default=None, choices=["pool_invariant", "c6_equivariant"])
    parser.add_argument("--gauge_anchor_mode", type=str, default=None, choices=["index", "geodesic", "tangent_max"])
    parser.add_argument("--gauge_debug", type=int, default=None)
    parser.add_argument("--downsample_mode", type=str, default="auto", choices=["auto", "current_default", "area_avg"])
    parser.add_argument("--upsample_mode", type=str, default="auto", choices=["auto", "current_default", "geodesic_kernel"])
    parser.add_argument("--upsample_sigma", type=float, default=None)
    parser.add_argument("--eq_loss_weight", type=float, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args, num_classes: int, device: torch.device, resolved_cfg: Dict[str, Any]) -> nn.Module:
    model = SO3UFormer(
        img_rank=resolved_cfg["img_rank"],
        node_type=resolved_cfg["node_type"],
        in_channels=3,
        out_channels=num_classes,
        in_scale_factor=2,
        num_scales=4,
        win_size_coef=2,
        enc_depths=2,
        dec_depths=2,
        bottleneck_depth=2,
        d_head_coef=2,
        enc_num_heads=[2, 4, 8, 16],
        dec_num_heads=[16, 16, 8, 4],
        bottleneck_num_heads=None,
        abs_pos_enc_in=True,
        abs_pos_enc=True,
        rel_pos_bias=True,
        rel_pos_bias_size=7,
        rel_pos_init_variance=1.0,
        downsample="center",
        upsample="interpolate",
        use_quadrature_attn=resolved_cfg["use_quadrature_attn"],
        quadrature_mode=resolved_cfg["quadrature_mode"],
        use_abs_phi_pe=resolved_cfg["use_abs_phi_pe"],
        rel_pos_bias_type=resolved_cfg["rel_pos_bias_type"],
        rel_pos_bins=resolved_cfg["rel_pos_bins"],
        gauge_num_frames=resolved_cfg["gauge_num_frames"],
        gauge_m_max=resolved_cfg["gauge_m_max"],
        gauge_mode=resolved_cfg["gauge_mode"],
        gauge_anchor_mode=resolved_cfg["gauge_anchor_mode"],
        gauge_debug=resolved_cfg["gauge_debug"],
        downsample_mode=resolved_cfg["downsample_mode"],
        upsample_mode=resolved_cfg["upsample_mode"],
        upsample_sigma=resolved_cfg["upsample_sigma"],
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        attn_out_drop_rate=0.0,
        pos_drop_rate=0.0,
        debug_skip_attn=False,
        append_self=False,
        use_checkpoint=True,
    )
    # Safety: mirror resolved config onto the model instance
    for key in (
        "use_quadrature_attn",
        "quadrature_mode",
        "use_abs_phi_pe",
        "rel_pos_bias_type",
        "rel_pos_bins",
        "gauge_num_frames",
        "gauge_m_max",
        "gauge_mode",
        "gauge_anchor_mode",
        "gauge_debug",
        "downsample_mode",
        "upsample_mode",
        "upsample_sigma",
    ):
        if hasattr(model, key):
            setattr(model, key, resolved_cfg[key])
    state = torch.load(args.weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    strict = bool(args.strict_load)
    if strict:
        model.load_state_dict(state, strict=True)
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print("[WARN] missing keys:", missing)
        if unexpected:
            print("[WARN] unexpected keys:", unexpected)
    model.to(device)
    model.eval()
    return model


def build_edges(rank: int, node_type: str, icosphere_ref: IcoSphereRef) -> torch.Tensor:
    neigh = icosphere_ref.get_neighbor_mapping(rank, depth=1)
    edges = []
    for i, nbrs in enumerate(neigh):
        for j in nbrs:
            if i < j:
                edges.append((i, j))
    return torch.tensor(edges, dtype=torch.long)


def build_positions(rank: int, node_type: str, icosphere_ref: IcoSphereRef) -> np.ndarray:
    normals = icosphere_ref.get_normals(rank=rank)
    return normals.astype(np.float64)


def build_tree(positions: np.ndarray) -> NearestNeighbors:
    tree = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(positions)
    return tree


def get_mapping(tree: NearestNeighbors, pos_out: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    # We want original features at R^{-1} * pos_out
    query = (rotation.T @ pos_out.T).T
    dist, idx = tree.kneighbors(query, return_distance=True)
    return idx.squeeze(1).astype(np.int64)


@torch.no_grad()
def evaluate_once(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mapping: np.ndarray,
    mask_label_zero: bool,
    edges: torch.Tensor,
    noise_std: float = 0.0,
    num_classes: int = None,
):
    cm = ConfusionMatrix(num_classes=num_classes)
    tp_edge = fp_edge = fn_edge = 0.0

    idx = torch.from_numpy(mapping).to(device)
    edges = edges.to(device)

    for batch in loader:
        rgb = batch["normalized_sphere_rgb"].to(device)  # (B, N_in, 3)
        gt = batch["sphere_gt_sem"].to(device)  # (B, N_in)
        mask = batch["sphere_valid_mask"].to(device).bool()

        rgb_rot = rgb.index_select(1, idx)
        gt_rot = gt.index_select(1, idx)
        mask_rot = mask.index_select(1, idx)

        if noise_std > 0:
            rgb_rot = rgb_rot + noise_std * torch.randn_like(rgb_rot)

        logits = model(rgb_rot)

        # Confusion matrix update (Evaluator logic)
        pred = logits.clone()
        pred[:, :, 0] = -float("inf")
        if mask_label_zero:
            pred = torch.where(gt_rot.unsqueeze(2) > 0, pred, torch.tensor(-float("inf"), device=device))
        pred_class = pred.argmax(dim=-1)
        gt_flat = gt_rot.view(-1).cpu().numpy()
        pred_flat = pred_class.view(-1).cpu().numpy()
        cm.confusion_matrix += compute_cm_numpy(gt_flat, pred_flat, cm.num_classes)

        # Boundary F1 on edges (edge-level)
        gt_u = gt_rot[:, edges[:, 0]]
        gt_v = gt_rot[:, edges[:, 1]]
        pred_u = pred_class[:, edges[:, 0]]
        pred_v = pred_class[:, edges[:, 1]]
        mask_u = mask_rot[:, edges[:, 0]]
        mask_v = mask_rot[:, edges[:, 1]]
        valid_edge = mask_u & mask_v & (gt_u > 0) & (gt_v > 0)
        gt_edge = (gt_u != gt_v) & valid_edge
        pred_edge = (pred_u != pred_v) & valid_edge

        tp_edge += (pred_edge & gt_edge).sum().item()
        fp_edge += (pred_edge & (~gt_edge)).sum().item()
        fn_edge += ((~pred_edge) & gt_edge).sum().item()

    errors = compute_segmentation_metrics(cm.confusion_matrix)
    bf1 = 0.0
    denom = (2 * tp_edge + fp_edge + fn_edge)
    if denom > 0:
        bf1 = 2 * tp_edge / denom
    errors["boundary_f1"] = bf1
    return errors


def compute_cm_numpy(gt_flat: np.ndarray, pred_flat: np.ndarray, num_classes: int) -> np.ndarray:
    # lightweight confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (gt_flat >= 0) & (gt_flat < num_classes) & (pred_flat >= 0) & (pred_flat < num_classes)
    gt_flat = gt_flat[valid]
    pred_flat = pred_flat[valid]
    np.add.at(cm, (gt_flat, pred_flat), 1)
    return cm


def run_group(
    name: str,
    model: nn.Module,
    dataset_root: str,
    dataset_name: str,
    eval_split: str,
    eval_rank: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    mapping_mats: List[np.ndarray],
    mask_label_zero: bool,
    edges: torch.Tensor,
    noise_std: float = 0.0,
    repeats: int = 1,
    seed: int = 123,
    num_classes: int = None,
    node_type: str = "vertex",
) -> Tuple[Dict[str, float], List[float]]:
    results = []
    miou_runs = []
    loader = build_loader(dataset_root, dataset_name, eval_split, eval_rank, batch_size, num_workers, node_type)
    for rep in range(repeats):
        set_seed(seed + rep)
        metrics_list = []
        for mapping in mapping_mats:
            metrics = evaluate_once(
                model=model,
                loader=loader,
                device=device,
                mapping=mapping,
                mask_label_zero=mask_label_zero,
                edges=edges,
                noise_std=noise_std,
                num_classes=num_classes,
            )
            metrics_list.append(metrics)
        # average over rotations for this repeat
        agg_rep = defaultdict(list)
        for m in metrics_list:
            for k, v in m.items():
                agg_rep[k].append(v)
        mean_rep = {k: float(np.mean(v)) for k, v in agg_rep.items()}
        results.append(mean_rep)
        miou_runs.append(mean_rep.get("acc/iou", 0.0))
    # aggregate across repeats
    agg = defaultdict(list)
    for r in results:
        for k, v in r.items():
            agg[k].append(v)
    mean_metrics = {k: float(np.mean(v)) for k, v in agg.items()}
    return mean_metrics, miou_runs


def build_loader(
    dataset_root: str,
    dataset_name: str,
    eval_split: str,
    sphere_rank: int,
    batch_size: int,
    num_workers: int,
    node_type: str,
) -> DataLoader:
    split_name = "val" if eval_split == "val" else "test"
    list_file = os.path.join(dataset_root, dataset_name, "splits_2d3d", f"stanford2d3d_{split_name}.txt")
    if not os.path.isfile(list_file):
        list_file = f"./data/splits_2d3d/stanford2d3d_{split_name}.txt"
    ds = Stanford2D3D(
        root_dir=dataset_root,
        list_file=list_file,
        dataset_kwargs={
            "sphere_rank": sphere_rank,
            "grid_width": 512,
            "sphere_node_type": node_type,
        },
        augmentation_kwargs=dict(
            color_augmentation=False,
            lr_flip_augmentation=False,
            yaw_rotation_augmentation=False,
        ),
        is_training=False,
        dataset_name=dataset_name,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )
    return loader


def resolve_eval_config(args) -> Dict[str, Any]:
    cfg = {
        "use_quadrature_attn": False,
        "quadrature_mode": "logit",
        "use_abs_phi_pe": True,
        "rel_pos_bias_type": "grid7_dtheta_dphi",
        "rel_pos_bins": 32,
        "gauge_num_frames": 3,
        "gauge_m_max": 2,
        "gauge_mode": "pool_invariant",
        "gauge_anchor_mode": "tangent_max",
        "gauge_debug": 0,
        "downsample_mode": "current_default",
        "upsample_mode": "current_default",
        "upsample_sigma": 0.4,
        "img_rank": 7,
        "node_type": "vertex",
        "eq_loss_weight": 0.0,
    }

    weights_path = os.path.abspath(args.weights_path)
    models_dir = os.path.dirname(weights_path)
    if os.path.basename(models_dir) == "models":
        exp_dir = os.path.dirname(models_dir)
    else:
        parts = weights_path.split(os.sep)
        if "models" in parts:
            idx = len(parts) - 1 - parts[::-1].index("models")
            exp_dir = os.sep.join(parts[:idx]) or os.sep
        else:
            exp_dir = os.path.dirname(weights_path)
    metadata: Dict[str, Any] = {}
    state_dict: Optional[Dict[str, Any]] = None
    try:
        ckpt = torch.load(args.weights_path, map_location="cpu")
        if isinstance(ckpt, dict):
            for key in ("args", "hparams", "config"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    metadata.update(ckpt[key])
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state_dict = ckpt["state_dict"]
            else:
                # support plain state_dict
                state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    except Exception:
        state_dict = None

    for name in ("args.json", "config.json", "hparams.yaml", "config.yaml", "train_args.json"):
        p = os.path.join(exp_dir, name)
        if os.path.isfile(p):
            try:
                if p.endswith(".json"):
                    with open(p, "r") as f:
                        metadata.update(json.load(f))
                else:
                    with open(p, "r") as f:
                        metadata.update(yaml.safe_load(f) or {})
            except Exception:
                pass

    # unwrap nested config blocks if present
    for parent_key in ("model", "train"):
        if isinstance(metadata.get(parent_key), dict):
            for k, v in metadata[parent_key].items():
                if k not in metadata:
                    metadata[k] = v

    # Parse flags from train log if present
    for train_log in (os.path.join(exp_dir, "train", "train.log"), os.path.join(exp_dir, "train.log")):
        if os.path.isfile(train_log):
            try:
                with open(train_log, "r") as f:
                    for line in f:
                        if line.strip().startswith("[FLAGS]"):
                            parts = shlex.split(line.strip().replace("[FLAGS]", "", 1))
                            for i in range(0, len(parts), 2):
                                if parts[i].startswith("--") and i + 1 < len(parts):
                                    key = parts[i][2:]
                                    metadata[key] = parts[i + 1]
                            break
                break
            except Exception:
                pass

    def infer_from_state(key: str):
        if state_dict is None:
            return None
        keys = list(state_dict.keys())
        if key == "use_abs_phi_pe":
            has_abs = any(("q_abs_pos_proj" in k or "k_abs_pos_proj" in k or "abs_pos_enc_in.1.weight" in k) for k in keys)
            return bool(has_abs)
        if key == "rel_pos_bias_type":
            if any("rel_pos_bias.gauge_A" in k or "rel_pos_bias.gauge_B" in k for k in keys):
                return "gauge_pool"
            if any("rel_pos_bias.bias_1d" in k for k in keys):
                return "geodesic_1d"
            return "grid7_dtheta_dphi"
        if key == "rel_pos_bins":
            if any("rel_pos_bias.gauge_A" in k for k in keys):
                for k in keys:
                    if "rel_pos_bias.gauge_A" in k:
                        return int(state_dict[k].shape[-1])
            if any("rel_pos_bias.bias_1d" in k for k in keys):
                for k in keys:
                    if "rel_pos_bias.bias_1d" in k:
                        return int(state_dict[k].shape[-1])
        if key == "gauge_m_max":
            for k in keys:
                if "rel_pos_bias.gauge_A" in k:
                    return int(state_dict[k].shape[1] - 1)
        return None

    def infer_from_run_name(key: str):
        run_name = os.path.basename(exp_dir)
        if key == "downsample_mode":
            if "__C-downAreaAvg" in run_name or "__Cboth__" in run_name:
                return "area_avg"
            if "__C0__" in run_name:
                return "current_default"
        if key == "upsample_mode":
            if "__C-upGeoKernel" in run_name or "__Cboth__" in run_name or "upGeoKernel" in run_name:
                return "geodesic_kernel"
            if "__C0__" in run_name:
                return "current_default"
        if key == "upsample_sigma":
            import re
            m = re.search(r"(?:upGeoKernelS|CbothS)([0-9]+p[0-9]+)", run_name)
            if m:
                return float(m.group(1).replace("p", "."))
        return None

    resolved = {}
    for key in cfg.keys():
        cli_val = getattr(args, key, None)
        if cli_val is not None and cli_val != "auto":
            resolved[key] = cli_val
            continue
        if key in metadata:
            resolved[key] = metadata[key]
            continue
        inferred = infer_from_state(key)
        if inferred is not None:
            resolved[key] = inferred
            print(f"[WARN] missing config '{key}', inferred from checkpoint: {inferred}")
            continue
        inferred = infer_from_run_name(key)
        if inferred is not None:
            resolved[key] = inferred
            print(f"[WARN] missing config '{key}', inferred from run name: {inferred}")
            continue
        resolved[key] = cfg[key]
        print(f"[WARN] missing config '{key}', using default {cfg[key]}")

    def parse_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return bool(value)
        if isinstance(value, str):
            if value.lower() in ("1", "true", "yes", "y", "t"):
                return True
            if value.lower() in ("0", "false", "no", "n", "f"):
                return False
        return bool(value)

    resolved["use_quadrature_attn"] = parse_bool(resolved["use_quadrature_attn"])
    resolved["use_abs_phi_pe"] = parse_bool(resolved["use_abs_phi_pe"])
    resolved["rel_pos_bins"] = int(resolved["rel_pos_bins"])
    resolved["gauge_num_frames"] = int(resolved["gauge_num_frames"])
    resolved["gauge_m_max"] = int(resolved["gauge_m_max"])
    resolved["gauge_debug"] = int(parse_bool(resolved["gauge_debug"]))
    resolved["img_rank"] = int(resolved["img_rank"])
    resolved["upsample_sigma"] = float(resolved["upsample_sigma"])
    resolved["eq_loss_weight"] = float(resolved["eq_loss_weight"])
    resolved["node_type"] = str(resolved["node_type"])
    resolved["quadrature_mode"] = "value_renorm" if str(resolved["quadrature_mode"]) == "value" else str(resolved["quadrature_mode"])
    resolved["gauge_mode"] = str(resolved["gauge_mode"])
    resolved["gauge_anchor_mode"] = str(resolved["gauge_anchor_mode"])

    print("Resolved eval config:")
    for k in cfg.keys():
        print(f"  {k}: {resolved[k]}")

    return resolved


def generate_rotations(group: str, full3d_samples: int) -> List[Rotation]:
    rots = []
    if group == "horizontal":
        for deg in range(0, 360, 45):
            rots.append(Rotation.from_euler("z", deg, degrees=True))
    elif group == "vertical":
        for deg in range(0, 180, 30):
            rots.append(Rotation.from_euler("x", deg, degrees=True))
    elif group == "full3d":
        for _ in range(full3d_samples):
            angles = np.random.uniform(low=[0, 0, 0], high=[360, 180, 360])
            rots.append(Rotation.from_euler("zyx", angles, degrees=True))
    else:
        rots.append(Rotation.identity())
    return rots


def build_mappings_for_group(rots: List[Rotation], tree_in: NearestNeighbors, pos_out: np.ndarray) -> List[np.ndarray]:
    mappings = []
    for r in rots:
        mapping = get_mapping(tree_in, pos_out, r.as_matrix())
        mappings.append(mapping)
    return mappings


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    resolved_cfg = resolve_eval_config(args)

    # Prepare geometry
    icosphere_ref = IcoSphereRef(node_type=resolved_cfg["node_type"])
    pos_out = build_positions(resolved_cfg["img_rank"], resolved_cfg["node_type"], icosphere_ref)
    tree_out = build_tree(pos_out)  # not used but handy
    edges = build_edges(resolved_cfg["img_rank"], resolved_cfg["node_type"], icosphere_ref)

    # Build baseline dataset to get num_classes (avoid first-batch max which can be incomplete)
    baseline_loader = build_loader(
        args.dataset_root_dir,
        args.dataset_name,
        args.eval_split,
        resolved_cfg["img_rank"],
        args.batch_size,
        args.num_workers,
        resolved_cfg["node_type"],
    )
    num_classes = int(getattr(baseline_loader.dataset, "NUM_CLASSES"))

    model = build_model(args, num_classes, device, resolved_cfg)

    # Baseline (no rotation)
    tree_in_baseline = build_tree(build_positions(resolved_cfg["img_rank"], resolved_cfg["node_type"], icosphere_ref))
    baseline_mapping = get_mapping(tree_in_baseline, pos_out, np.eye(3))
    baseline_metrics, baseline_mious = run_group(
        name="baseline",
        model=model,
        dataset_root=args.dataset_root_dir,
        dataset_name=args.dataset_name,
        eval_split=args.eval_split,
        eval_rank=resolved_cfg["img_rank"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        mapping_mats=[baseline_mapping],
        mask_label_zero=True,
        edges=edges,
        noise_std=0.0,
        repeats=args.repeats,
        seed=args.seed,
        num_classes=num_classes,
        node_type=resolved_cfg["node_type"],
    )

    results_summary = {"baseline": baseline_metrics}
    miou_runs = {"baseline": baseline_mious}

    if args.baseline_only:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(args.log_dir, exist_ok=True)
        json_path = os.path.join(args.log_dir, f"rotation_sensitivity_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump({"results": results_summary, "deltas": {}, "t_tests": {}}, f, indent=2)
        print(f"Saved results to {json_path}")
        return

    # Horizontal
    horiz_rots = generate_rotations("horizontal", args.full3d_samples)
    horiz_maps = build_mappings_for_group(horiz_rots, tree_in_baseline, pos_out)
    results_summary["horizontal"], miou_runs["horizontal"] = run_group(
        "horizontal",
        model,
        args.dataset_root_dir,
        args.dataset_name,
        args.eval_split,
        resolved_cfg["img_rank"],
        args.batch_size,
        args.num_workers,
        device,
        horiz_maps,
        True,
        edges,
        noise_std=0.0,
        repeats=args.repeats,
        seed=args.seed,
        num_classes=num_classes,
        node_type=resolved_cfg["node_type"],
    )

    # Vertical (tilt)
    vert_rots = generate_rotations("vertical", args.full3d_samples)
    vert_maps = build_mappings_for_group(vert_rots, tree_in_baseline, pos_out)
    results_summary["vertical"], miou_runs["vertical"] = run_group(
        "vertical",
        model,
        args.dataset_root_dir,
        args.dataset_name,
        args.eval_split,
        resolved_cfg["img_rank"],
        args.batch_size,
        args.num_workers,
        device,
        vert_maps,
        True,
        edges,
        noise_std=0.0,
        repeats=args.repeats,
        seed=args.seed,
        num_classes=num_classes,
        node_type=resolved_cfg["node_type"],
    )

    # Full 3D
    full_rots = generate_rotations("full3d", args.full3d_samples)
    full_maps = build_mappings_for_group(full_rots, tree_in_baseline, pos_out)
    results_summary["full3d"], miou_runs["full3d"] = run_group(
        "full3d",
        model,
        args.dataset_root_dir,
        args.dataset_name,
        args.eval_split,
        resolved_cfg["img_rank"],
        args.batch_size,
        args.num_workers,
        device,
        full_maps,
        True,
        edges,
        noise_std=0.0,
        repeats=args.repeats,
        seed=args.seed,
        num_classes=num_classes,
        node_type=resolved_cfg["node_type"],
    )

    # Noise group (full3d rotations + gaussian noise)
    results_summary["noise"], miou_runs["noise"] = run_group(
        "noise",
        model,
        args.dataset_root_dir,
        args.dataset_name,
        args.eval_split,
        resolved_cfg["img_rank"],
        args.batch_size,
        args.num_workers,
        device,
        full_maps,
        True,
        edges,
        noise_std=args.noise_std,
        repeats=args.repeats,
        seed=args.seed,
        num_classes=num_classes,
        node_type=resolved_cfg["node_type"],
    )

    # Multi-resolution group (evaluate rank list mapped to model rank)
    eval_ranks = [int(r.strip()) for r in args.eval_ranks.split(",") if r.strip()]
    multi_res_metrics = {}
    for rnk in eval_ranks:
        pos_in = build_positions(rnk, resolved_cfg["node_type"], icosphere_ref)
        tree_in = build_tree(pos_in)
        maps = build_mappings_for_group([Rotation.identity()], tree_in, pos_out)
        metrics_rank, miou_rank = run_group(
            f"rank_{rnk}",
            model,
            args.dataset_root_dir,
            args.dataset_name,
            args.eval_split,
            rnk,
            args.batch_size,
            args.num_workers,
            device,
            maps,
            True,
            edges,
            noise_std=0.0,
            repeats=args.repeats,
            seed=args.seed,
            num_classes=num_classes,
            node_type=resolved_cfg["node_type"],
        )
        multi_res_metrics[f"rank_{rnk}"] = metrics_rank
        miou_runs[f"rank_{rnk}"] = miou_rank
    results_summary["multi_resolution"] = multi_res_metrics

    # Compute deltas and t-tests for mIoU
    baseline_miou = results_summary["baseline"]["acc/iou"]
    deltas = {}
    ttests = {}
    for k, v in results_summary.items():
        if k == "baseline":
            continue
        if isinstance(v, dict) and "acc/iou" in v:
            deltas[k] = baseline_miou - v["acc/iou"]
            stat, pval = stats.ttest_ind(miou_runs["baseline"], miou_runs[k], equal_var=False)
            ttests[k] = {"t_stat": float(stat), "p_value": float(pval)}
        elif isinstance(v, dict):
            # multi_res nested
            deltas[k] = {kk: baseline_miou - vv["acc/iou"] for kk, vv in v.items()}
            for kk in v.keys():
                stat, pval = stats.ttest_ind(miou_runs["baseline"], miou_runs[kk], equal_var=False)
                ttests[f"{k}_{kk}"] = {"t_stat": float(stat), "p_value": float(pval)}

    # Save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.log_dir, exist_ok=True)
    json_path = os.path.join(args.log_dir, f"rotation_sensitivity_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump({"results": results_summary, "deltas": deltas, "t_tests": ttests}, f, indent=2)

    # Plot ΔmIoU bars for main groups
    groups = ["horizontal", "vertical", "full3d", "noise"]
    delta_vals = [deltas[g] for g in groups]
    plt.figure(figsize=(8, 4))
    plt.bar(groups, delta_vals, color="#4C72B0")
    plt.ylabel("Δ mIoU (baseline - group)")
    plt.title("Rotation sensitivity (lower is better)")
    plt.tight_layout()
    plt_path = os.path.join(args.log_dir, f"rotation_sensitivity_{timestamp}.png")
    plt.savefig(plt_path)
    print(f"Saved results to {json_path}")
    print(f"Saved plot to {plt_path}")


if __name__ == "__main__":
    main()
