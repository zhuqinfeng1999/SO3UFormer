import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from tqdm import tqdm

from trimesh_utils import IcoSphereRef, asSpherical

_OMEGA_STATS_PRINTED = set()


def get_rotation_matrices(rphitheta: np.ndarray) -> np.ndarray:
    r, p, t = rphitheta.T
    p = (90-p) / 180 * np.pi
    t = t / 180 * np.pi
    zeros, ones = np.zeros_like(t), np.ones_like(t)
    mat_t = np.array([
        [np.cos(-t), -np.sin(-t), zeros],
        [np.sin(-t), np.cos(-t), zeros],
        [zeros, zeros, ones],
    ]).transpose(2,0,1)
    mat_p = np.array([
        [np.cos(p), zeros, np.sin(p)],
        [zeros, ones, zeros],
        [-np.sin(p), zeros, np.cos(p)],
    ]).transpose(2,0,1)
    return mat_p @ mat_t


class GlobalVerticalPositionEnconding(nn.Module):
    def __init__(self, rank: int, icosphere_ref: IcoSphereRef, mode: str, num_pos_feats, max_frequency, min_frequency=1, scale=None):
        super().__init__()

        self.num_pos_feats = num_pos_feats
        self.max_frequency = max_frequency
        self.min_frequency = min_frequency
        if scale is None:
            scale = math.pi
        self.scale = scale

        normals = icosphere_ref.get_normals(rank)

        # pos_value in [-1, 1]
        if mode == "z":
            pos_value = normals[:, 2]
        elif mode == "phi":
            pos_value = asSpherical(normals)[:, 1] / 180 * 2 - 1
        else:
            raise NotImplementedError()
        pos_value = torch.tensor(pos_value.copy(), dtype=torch.float32).view(-1,1)

        dim_f = torch.linspace(0, 1, num_pos_feats // 2, dtype=torch.float32)
        dim_f = self.max_frequency ** dim_f * self.min_frequency ** (1-dim_f)

        pos = pos_value * self.scale * dim_f
        pos = torch.stack((pos.sin(), pos.cos()), dim=1).flatten(start_dim=1)
        self.register_buffer("pos", pos, persistent=False)

    def forward(self, x: Tensor):
        # N, D, C = x.shape
        # assert D == self.pos.shape[0]
        pos = self.pos.unsqueeze(0)
        return pos


class RelativePositionBias(nn.Module):
    def __init__(
            self,
            rank: int,
            icosphere_ref: IcoSphereRef,
            win_size_coef: int,
            rel_pos_bias_size: int,
            num_heads: int,
            init_variance: float = 10,
            use_quadrature_attn: bool = False,
            rel_pos_bias_type: str = "grid7_dtheta_dphi",
            rel_pos_bins: int = 32,
            gauge_num_frames: int = 3,
            gauge_m_max: int = 2,
            gauge_mode: str = "pool_invariant",
            gauge_anchor_mode: str = "tangent_max",
            gauge_debug: bool = False,
    ):
        assert rel_pos_bias_size > 0

        super().__init__()

        self.rank = rank
        self.use_quadrature_attn = use_quadrature_attn
        self.rel_pos_bias_type = rel_pos_bias_type
        self.rel_pos_bins = rel_pos_bins
        self.gauge_num_frames = gauge_num_frames
        self.gauge_m_max = gauge_m_max
        self.gauge_mode = gauge_mode
        self.gauge_anchor_mode = gauge_anchor_mode
        self.gauge_debug = gauge_debug
        self._gauge_debug_printed = False

        normals = icosphere_ref.get_normals(rank)

        mapping = icosphere_ref.get_neighbor_mapping(rank=rank, depth=win_size_coef)

        self.num_nodes = len(mapping)
        self.num_keys = max(len(_) for _ in mapping)

        # B, H, D, K, C_H (idx is the same for B,H,C_)
        idx = torch.arange(0, self.num_nodes).unsqueeze(1).expand(-1, self.num_keys).clone()  # each query by default keys itself
        idx_mask = torch.zeros(self.num_nodes, self.num_keys).bool()
        for i, keys in tqdm(enumerate(mapping), desc=f"RelativePositionBias - index mapping {rank}"):
            idx[i, :len(keys)] = torch.tensor(list(keys))
            idx_mask[i, :len(keys)] = 1

        # Register for use by the attention module
        self.register_buffer("idx", idx[None, None, :, :, None], persistent=False)
        self.register_buffer("idx_mask", idx_mask[None, None, :, :], persistent=False)

        if self.rel_pos_bias_type == "grid7_dtheta_dphi":
            normals_rphitheta = asSpherical(normals)
            rot_mat = get_rotation_matrices(normals_rphitheta)

            expanded_normals = torch.tensor(normals.copy(), dtype=torch.float64).unsqueeze(1).expand(-1, self.num_keys, -1)
            expanded_idx = idx.unsqueeze(2).expand(-1, -1, 3)
            aligned_neighbors = torch.gather(expanded_normals, dim=0, index=expanded_idx).numpy()

            rotated_neighbors = (rot_mat @ aligned_neighbors.transpose(0,2,1)).transpose(0,2,1)
            rotated_neighbors_flat = rotated_neighbors.reshape(self.num_nodes*self.num_keys, 3)
            relative_coords_flat = (asSpherical(rotated_neighbors_flat)[:, 1:] - np.array([[90, 0]]))
            relative_coords = relative_coords_flat.reshape(self.num_nodes, self.num_keys, 2)

            self.register_buffer("relative_coords", torch.tensor(relative_coords).float(), persistent=False)
            self.bias_grid = nn.Parameter(init_variance * torch.randn(1, num_heads, rel_pos_bias_size, rel_pos_bias_size), requires_grad=True)
        elif self.rel_pos_bias_type == "geodesic_1d":
            eps = 1e-6
            expanded_normals = torch.tensor(normals.copy(), dtype=torch.float64).unsqueeze(1).expand(-1, self.num_keys, -1)
            expanded_idx = idx.unsqueeze(2).expand(-1, -1, 3)
            aligned_neighbors = torch.gather(expanded_normals, dim=0, index=expanded_idx)
            cos_sim = (expanded_normals * aligned_neighbors).sum(-1).clamp(-1.0 + eps, 1.0 - eps)
            delta = torch.arccos(cos_sim)
            delta_norm = (delta / math.pi).float()
            self.register_buffer("delta_norm", delta_norm, persistent=False)
            self.bias_1d = nn.Parameter(torch.zeros(num_heads, rel_pos_bins), requires_grad=True)
        elif self.rel_pos_bias_type == "gauge_pool":
            if self.gauge_mode != "pool_invariant":
                raise NotImplementedError(f"Unsupported gauge_mode {self.gauge_mode} for gauge_pool")
            if self.gauge_anchor_mode == "index":
                warnings.warn("gauge_anchor_mode=index is not intrinsic; switching to tangent_max")
                self.gauge_anchor_mode = "tangent_max"
            eps = 1e-6
            expanded_normals = torch.tensor(normals.copy(), dtype=torch.float64).unsqueeze(1).expand(-1, self.num_keys, -1)
            expanded_idx = idx.unsqueeze(2).expand(-1, -1, 3)
            aligned_neighbors = torch.gather(expanded_normals, dim=0, index=expanded_idx)
            cos_sim = (expanded_normals * aligned_neighbors).sum(-1).clamp(-1.0 + eps, 1.0 - eps)
            delta = torch.arccos(cos_sim)
            delta_norm = (delta / math.pi).float()
            self.register_buffer("delta_norm", delta_norm, persistent=False)

            # build intrinsic local frames from anchor neighbors (topology order or geometry-aware)
            anchor_idx = self._build_anchor_indices(
                mapping,
                self.num_nodes,
                self.gauge_num_frames,
                normals,
                self.gauge_anchor_mode,
            )
            normals_tensor = torch.tensor(normals.copy(), dtype=torch.float64)
            anchor_xyz = normals_tensor[anchor_idx]
            n = normals_tensor
            n = n / n.norm(dim=-1, keepdim=True).clamp(min=eps)
            n_expand = n.unsqueeze(1).expand(-1, self.gauge_num_frames, -1)
            proj = anchor_xyz - (anchor_xyz * n_expand).sum(-1, keepdim=True) * n_expand
            proj_norm = proj.norm(dim=-1, keepdim=True).clamp(min=eps)
            e1 = proj / proj_norm
            e2 = torch.cross(n_expand, e1, dim=-1)

            t = aligned_neighbors - (aligned_neighbors * n.unsqueeze(1)).sum(-1, keepdim=True) * n.unsqueeze(1)
            t = t.unsqueeze(1).expand(-1, self.gauge_num_frames, -1, -1)
            e1_expand = e1.unsqueeze(2)
            e2_expand = e2.unsqueeze(2)
            t_dot_e1 = (t * e1_expand).sum(-1)
            t_dot_e2 = (t * e2_expand).sum(-1)
            alpha = torch.atan2(t_dot_e2, t_dot_e1)
            alpha = alpha * idx_mask.unsqueeze(1).float()
            self.register_buffer("alpha", alpha.permute(1, 0, 2).to(torch.float16), persistent=False)

            self.gauge_A = nn.Parameter(torch.zeros(num_heads, self.gauge_m_max + 1, rel_pos_bins), requires_grad=True)
            self.gauge_B = nn.Parameter(torch.zeros(num_heads, self.gauge_m_max + 1, rel_pos_bins), requires_grad=True)

            if self.gauge_debug:
                n_norm = n.norm(dim=-1)
                e1_norm = e1.norm(dim=-1)
                e2_norm = e2.norm(dim=-1)
                alpha_valid = alpha[idx_mask.unsqueeze(1)]
                self._gauge_debug_stats = dict(
                    n_norm_min=float(n_norm.min().item()),
                    n_norm_mean=float(n_norm.mean().item()),
                    e1_norm_min=float(e1_norm.min().item()),
                    e1_norm_mean=float(e1_norm.mean().item()),
                    e2_norm_min=float(e2_norm.min().item()),
                    e2_norm_mean=float(e2_norm.mean().item()),
                    alpha_min=float(alpha_valid.min().item()) if alpha_valid.numel() else 0.0,
                    alpha_max=float(alpha_valid.max().item()) if alpha_valid.numel() else 0.0,
                    alpha_mean=float(alpha_valid.mean().item()) if alpha_valid.numel() else 0.0,
                    alpha_std=float(alpha_valid.std().item()) if alpha_valid.numel() else 0.0,
                )
        else:
            raise ValueError(f"Unsupported rel_pos_bias_type {self.rel_pos_bias_type}")

        if self.use_quadrature_attn:
            omega = self._compute_quadrature_weights(icosphere_ref, rank)
            omega = omega / omega.mean()
            omega_neighbors = omega[idx]
            omega_neighbors = torch.where(idx_mask, omega_neighbors, torch.ones_like(omega_neighbors))
            self.register_buffer("omega_neighbors", omega_neighbors[None, None, :, :], persistent=False)

            key = (icosphere_ref.node_type, rank)
            if key not in _OMEGA_STATS_PRINTED:
                _OMEGA_STATS_PRINTED.add(key)
                omega_min = float(omega.min().item())
                omega_max = float(omega.max().item())
                omega_mean = float(omega.mean().item())
                print(f"[Quadrature] omega stats (node_type={icosphere_ref.node_type}, rank={rank}) "
                      f"min={omega_min:.6f} max={omega_max:.6f} mean={omega_mean:.6f}")

    def _compute_quadrature_weights(self, icosphere_ref: IcoSphereRef, rank: int) -> torch.Tensor:
        ico = icosphere_ref.get_icosphere(rank, refine=True)
        if icosphere_ref.node_type == "face":
            omega = np.asarray(ico.area_faces, dtype=np.float64)
        elif icosphere_ref.node_type == "vertex":
            faces = np.asarray(ico.faces, dtype=np.int64)
            face_areas = np.asarray(ico.area_faces, dtype=np.float64)
            omega = np.zeros((ico.vertices.shape[0],), dtype=np.float64)
            contrib = face_areas / 3.0
            np.add.at(omega, faces[:, 0], contrib)
            np.add.at(omega, faces[:, 1], contrib)
            np.add.at(omega, faces[:, 2], contrib)
        else:
            raise NotImplementedError(f"Unsupported node type {icosphere_ref.node_type}")
        return torch.tensor(omega, dtype=torch.float32)

    def get_neighbor_idx(self):
        return self.idx, self.idx_mask

    def _build_anchor_indices(self, mapping, num_nodes: int, num_frames: int, normals, mode: str) -> torch.Tensor:
        anchor_idx = []
        normals = np.asarray(normals, dtype=np.float64)
        for i in range(num_nodes):
            neighbors = list(mapping[i])
            neighbors = [n for n in neighbors if n != i]
            if not neighbors:
                neighbors = [i]
            if mode == "index":
                warnings.warn("gauge_anchor_mode=index is not intrinsic; switching to tangent_max")
                mode = "tangent_max"
            n_i = normals[i]
            cand_xyz = normals[neighbors]
            dots = np.clip(cand_xyz @ n_i, -1.0, 1.0)
            if mode == "geodesic":
                order = np.argsort(-dots)  # closest first
            elif mode == "tangent_max":
                t_norm = np.sqrt(np.maximum(1.0 - dots ** 2, 0.0))
                order = np.argsort(-t_norm)
            else:
                raise ValueError(f"Unsupported gauge_anchor_mode {mode}")
            candidates = [neighbors[j] for j in order]
            picks = candidates[:num_frames]
            if len(picks) < num_frames:
                picks.extend([picks[-1]] * (num_frames - len(picks)))
            anchor_idx.append(picks)
        return torch.tensor(anchor_idx, dtype=torch.long)

    def forward(self, keys: Tensor):
        N, H, D, K, C_H = keys.shape
        if self.rel_pos_bias_type == "grid7_dtheta_dphi":
            assert D == self.relative_coords.shape[0]
            assert K == self.relative_coords.shape[1]
            rel_coords = self.relative_coords.unsqueeze(0)
            rel_coords_normalized = rel_coords / (rel_coords.abs().max() + 1e-8)
            rel_bias = F.grid_sample(self.bias_grid, grid=rel_coords_normalized, align_corners=True)
            return rel_coords, rel_bias

        if self.rel_pos_bias_type == "geodesic_1d":
            assert D == self.delta_norm.shape[0]
            assert K == self.delta_norm.shape[1]
            delta = self.delta_norm.to(keys.device)
            idx_float = delta * (self.rel_pos_bins - 1)
            idx0 = torch.floor(idx_float).long()
            idx1 = torch.clamp(idx0 + 1, max=self.rel_pos_bins - 1)
            w = (idx_float - idx0.float()).view(-1)
            idx0_flat = idx0.view(-1)
            idx1_flat = idx1.view(-1)
            bias_table = self.bias_1d.to(keys.device)
            bias0 = bias_table[:, idx0_flat]
            bias1 = bias_table[:, idx1_flat]
            bias = bias0 * (1.0 - w) + bias1 * w
            bias = bias.view(1, -1, D, K)
            return None, bias

        if self.rel_pos_bias_type == "gauge_pool":
            assert D == self.delta_norm.shape[0]
            assert K == self.delta_norm.shape[1]
            delta = self.delta_norm.to(keys.device)
            idx_float = delta * (self.rel_pos_bins - 1)
            idx0 = torch.floor(idx_float).long()
            idx1 = torch.clamp(idx0 + 1, max=self.rel_pos_bins - 1)
            w = (idx_float - idx0.float())
            idx0_flat = idx0.view(-1)
            idx1_flat = idx1.view(-1)

            A_table = self.gauge_A.to(keys.device)
            B_table = self.gauge_B.to(keys.device)
            A0 = A_table[:, :, idx0_flat].view(-1, self.gauge_m_max + 1, D, K)
            A1 = A_table[:, :, idx1_flat].view(-1, self.gauge_m_max + 1, D, K)
            B0 = B_table[:, :, idx0_flat].view(-1, self.gauge_m_max + 1, D, K)
            B1 = B_table[:, :, idx1_flat].view(-1, self.gauge_m_max + 1, D, K)
            w = w.view(1, 1, D, K)
            A = A0 * (1.0 - w) + A1 * w
            B = B0 * (1.0 - w) + B1 * w

            alpha = self.alpha.to(keys.device).to(torch.float32)
            if self.gauge_debug and not self._gauge_debug_printed:
                self._gauge_debug_printed = True
                stats = getattr(self, "_gauge_debug_stats", {})
                if stats:
                    print("[GaugeDebug] n_norm_min={n_norm_min:.6f} n_norm_mean={n_norm_mean:.6f} "
                          "e1_norm_min={e1_norm_min:.6f} e1_norm_mean={e1_norm_mean:.6f} "
                          "e2_norm_min={e2_norm_min:.6f} e2_norm_mean={e2_norm_mean:.6f} "
                          "alpha_min={alpha_min:.6f} alpha_max={alpha_max:.6f} "
                          "alpha_mean={alpha_mean:.6f} alpha_std={alpha_std:.6f}".format(**stats))
            num_rot = 6
            rot = 2.0 * math.pi * torch.arange(num_rot, device=alpha.device).view(-1, 1, 1, 1) / num_rot
            alpha_shifted = alpha.unsqueeze(0) - rot
            m = torch.arange(self.gauge_m_max + 1, device=alpha.device).view(-1, 1, 1, 1, 1)
            cos_ma = torch.cos(m * alpha_shifted.unsqueeze(0))
            sin_ma = torch.sin(m * alpha_shifted.unsqueeze(0))
            A_exp = A.unsqueeze(2).unsqueeze(2)
            B_exp = B.unsqueeze(2).unsqueeze(2)
            bias = (A_exp * cos_ma.unsqueeze(0) + B_exp * sin_ma.unsqueeze(0)).sum(dim=1)
            bias = bias.mean(dim=2).mean(dim=1)
            bias = bias * self.idx_mask.to(bias.device).squeeze(0).squeeze(0).unsqueeze(0)
            return None, bias.unsqueeze(0)

        raise ValueError(f"Unsupported rel_pos_bias_type {self.rel_pos_bias_type}")
