"""
1. 保存原始 MHR 参数作为 init buffer
2. 为当前 split 建立可学习 delta
3. 每次 forward(local_idx)：
   refined MHR params → MHR model forward → mesh_verts
4. 返回 RMAvatar 兼容的 mesh_info
5. 导出 full-length model_params_refined.npz
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.mhr_avatar_reader import (
    _resolve_mhr_model_file,
    _load_faces_from_mhr_torchscript,
    _mhr_to_rmavatar_space,
)


def compute_vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    verts: (B, V, 3)
    faces: (F, 3)
    return: (B, V, 3)
    """
    faces = faces.long()
    tris = verts[:, faces]  # (B, F, 3, 3)

    face_normals = torch.cross(
        tris[:, :, 1] - tris[:, :, 0],
        tris[:, :, 2] - tris[:, :, 0],
        dim=-1,
    )
    face_normals = F.normalize(face_normals, dim=-1, eps=1.0e-8)

    normals = torch.zeros_like(verts)

    for corner in range(3):
        normals.index_add_(1, faces[:, corner], face_normals)

    normals = F.normalize(normals, dim=-1, eps=1.0e-8)
    return normals


class MHRPoseRefinementModule(nn.Module):
    """
    Pose-only refinement module for mhravatar.

    First safe mode:
        optimize_pred_cam_t = True
        optimize_model_parameters = False
        optimize_expression = False

    Later optional mode:
        optimize_model_parameters = True
    """

    def __init__(
        self,
        frameset,
        optimize_model_parameters=False,
        optimize_expression=False,
        optimize_pred_cam_t=True,
        optimize_extra_transl=True,
        device="cuda",
    ):
        super().__init__()

        self.device = torch.device(device)
        self.frameset = frameset
        self.use_mhr_coord_fix = bool(frameset.use_mhr_coord_fix)

        # Split-local frame ids.
        self.frm_list = list(frameset.frm_list)
        self.num_frames = len(self.frm_list)

        frame_ids = torch.tensor(self.frm_list, dtype=torch.long)
        self.register_buffer("frame_ids", frame_ids)

        # Load frozen MHR model.
        mhr_model_file = _resolve_mhr_model_file(frameset.config.mhr_model_path)
        self.mhr_model = torch.jit.load(str(mhr_model_file), map_location=self.device)
        self.mhr_model.eval()

        for p in self.mhr_model.parameters():
            p.requires_grad_(False)

        faces, _ = _load_faces_from_mhr_torchscript(self.mhr_model)
        self.register_buffer("faces", faces.to(self.device).long())

        # Full original arrays for export.
        full_model_parameters = frameset.model_parameters.detach().float().clone()
        full_expression = frameset.expression_coeffs.detach().float().clone()

        self.has_pred_cam_t = frameset.pred_cam_t is not None

        if self.has_pred_cam_t:
            full_pred_cam_t = frameset.pred_cam_t.detach().float().clone()
        else:
            full_pred_cam_t = torch.zeros(
                full_model_parameters.shape[0],
                3,
                dtype=torch.float32,
            )

        # Identity is usually one identity vector.
        identity_coeffs = frameset.identity_coeffs.detach().float().clone()

        self.register_buffer("full_model_parameters_init", full_model_parameters.to(self.device))
        self.register_buffer("full_expression_init", full_expression.to(self.device))
        self.register_buffer("full_pred_cam_t_init", full_pred_cam_t.to(self.device))
        self.register_buffer("identity_coeffs", identity_coeffs.to(self.device))

        # Split-local init tensors.
        split_model_parameters = full_model_parameters[frame_ids].to(self.device)
        split_expression = full_expression[frame_ids].to(self.device)
        split_pred_cam_t = full_pred_cam_t[frame_ids].to(self.device)

        self.register_buffer("model_parameters_init", split_model_parameters)
        self.register_buffer("expression_init", split_expression)
        self.register_buffer("pred_cam_t_init", split_pred_cam_t)

        self.delta_model_parameters = nn.Parameter(
            torch.zeros_like(self.model_parameters_init),
            requires_grad=optimize_model_parameters,
        )

        self.delta_expression = nn.Parameter(
            torch.zeros_like(self.expression_init),
            requires_grad=optimize_expression,
        )

        # If original pred_cam_t exists, this directly refines it.
        # If not, this behaves like an extra root translation.
        self.delta_pred_cam_t = nn.Parameter(
            torch.zeros_like(self.pred_cam_t_init),
            requires_grad=optimize_pred_cam_t or optimize_extra_transl,
        )

        # Optional norm clamps. None or <=0 means disabled.
        self.max_model_delta = None
        self.max_expression_delta = None
        self.max_pred_cam_t_delta = None

    def optimizable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _normalize_idx(self, idx):
        if not torch.is_tensor(idx):
            idx = torch.tensor([idx], dtype=torch.long, device=self.device)
        else:
            idx = idx.to(self.device).long().view(-1)
        return idx

    def _clamp_delta_by_norm(self, delta, max_norm):
        if max_norm is None:
            return delta

        max_norm = float(max_norm)

        if max_norm <= 0:
            return delta

        norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1.0e-8)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return delta * scale

    def get_refined_params(self, idx):
        idx = self._normalize_idx(idx)

        delta_model = self._clamp_delta_by_norm(
            self.delta_model_parameters[idx],
            self.max_model_delta,
        )
        delta_expr = self._clamp_delta_by_norm(
            self.delta_expression[idx],
            self.max_expression_delta,
        )
        delta_cam = self._clamp_delta_by_norm(
            self.delta_pred_cam_t[idx],
            self.max_pred_cam_t_delta,
        )

        model_parameters = self.model_parameters_init[idx] + delta_model
        expression = self.expression_init[idx] + delta_expr
        pred_cam_t = self.pred_cam_t_init[idx] + delta_cam

        return {
            "model_parameters": model_parameters,
            "expression_coeffs": expression,
            "pred_cam_t": pred_cam_t,
        }

    def _identity_for_batch(self, batch_size):
        identity = self.identity_coeffs

        if identity.ndim == 1:
            identity = identity[None].expand(batch_size, -1).contiguous()
        elif identity.ndim == 2 and identity.shape[0] == 1:
            identity = identity.expand(batch_size, -1).contiguous()
        elif identity.ndim == 2 and identity.shape[0] == batch_size:
            identity = identity.contiguous()
        else:
            raise ValueError(
                f"Unsupported identity_coeffs shape: {tuple(identity.shape)}"
            )

        return identity

    def forward(self, idx):
        idx = self._normalize_idx(idx)
        params = self.get_refined_params(idx)

        batch_size = idx.shape[0]
        identity = self._identity_for_batch(batch_size)

        vertices, _ = self.mhr_model(
            identity,
            params["model_parameters"],
            params["expression_coeffs"],
        )

        vertices = _mhr_to_rmavatar_space(vertices, self.use_mhr_coord_fix)
        vertices = vertices + params["pred_cam_t"][:, None, :]

        normals = compute_vertex_normals(vertices, self.faces)

        if batch_size != 1:
            raise NotImplementedError(
                "Current mhravatar training/refinement loop assumes batch_size=1."
            )

        return {
            "mesh_verts": vertices[0],
            "mesh_norms": normals[0],
            "mesh_faces": self.faces,
            "pose": params["model_parameters"][0],
            "mhr_identity": self.identity_coeffs,
            "mhr_expr": params["expression_coeffs"][0],
            "pred_cam_t": params["pred_cam_t"][0],
        }

    def regularization(self, idx, cfg):
        idx = self._normalize_idx(idx)

        losses = {}

        lambda_model = float(cfg.get("lambda_model_parameters", 0.0))
        lambda_expr = float(cfg.get("lambda_expression", 0.0))
        lambda_cam = float(cfg.get("lambda_pred_cam_t", 0.0))

        if lambda_model > 0:
            losses["mhr_model_prior"] = (
                self.delta_model_parameters[idx].pow(2).mean() * lambda_model
            )

        if lambda_expr > 0:
            losses["mhr_expr_prior"] = (
                self.delta_expression[idx].pow(2).mean() * lambda_expr
            )

        if lambda_cam > 0:
            losses["mhr_pred_cam_t_prior"] = (
                self.delta_pred_cam_t[idx].pow(2).mean() * lambda_cam
            )

        losses.update(self._smoothness_loss(idx, cfg))
        return losses

    def _smoothness_loss(self, idx, cfg):
        losses = {}

        lambda_model = float(cfg.get("lambda_smooth_model_parameters", 0.0))
        lambda_expr = float(cfg.get("lambda_smooth_expression", 0.0))
        lambda_cam = float(cfg.get("lambda_smooth_pred_cam_t", 0.0))

        if lambda_model == 0 and lambda_expr == 0 and lambda_cam == 0:
            return losses

        idx = idx.view(-1)

        neighbor_pairs = []

        valid_prev = idx > 0
        valid_next = idx < self.num_frames - 1

        if valid_prev.any():
            neighbor_pairs.append((idx[valid_prev], idx[valid_prev] - 1))

        if valid_next.any():
            neighbor_pairs.append((idx[valid_next], idx[valid_next] + 1))

        if not neighbor_pairs:
            return losses

        def smooth_delta(delta):
            terms = []
            for center, neigh in neighbor_pairs:
                terms.append((delta[center] - delta[neigh]).pow(2).mean())
            return torch.stack(terms).mean()

        if lambda_model > 0:
            losses["mhr_model_smooth"] = (
                smooth_delta(self.delta_model_parameters) * lambda_model
            )

        if lambda_expr > 0:
            losses["mhr_expr_smooth"] = (
                smooth_delta(self.delta_expression) * lambda_expr
            )

        if lambda_cam > 0:
            losses["mhr_pred_cam_t_smooth"] = (
                smooth_delta(self.delta_pred_cam_t) * lambda_cam
            )

        return losses

    @torch.no_grad()
    def export_full_npz(self, out_path):
        """
        Export full-length MHR params npz.

        It keeps all non-refined frames unchanged and replaces only the current
        split frame ids with refined values.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        idx = torch.arange(self.num_frames, device=self.device, dtype=torch.long)
        params = self.get_refined_params(idx)

        full_model = self.full_model_parameters_init.detach().clone()
        full_expr = self.full_expression_init.detach().clone()
        full_cam = self.full_pred_cam_t_init.detach().clone()

        frame_ids = self.frame_ids.long()

        full_model[frame_ids] = params["model_parameters"]
        full_expr[frame_ids] = params["expression_coeffs"]
        full_cam[frame_ids] = params["pred_cam_t"]

        save_dict = {
            "model_parameters": full_model.detach().cpu().numpy().astype(np.float32),
            "identity_coeffs": self.identity_coeffs.detach().cpu().numpy().astype(np.float32),
            "expression_coeffs": full_expr.detach().cpu().numpy().astype(np.float32),
        }

        # Preserve pred_cam_t if original file had it, or if we optimized it as extra translation.
        save_dict["pred_cam_t"] = full_cam.detach().cpu().numpy().astype(np.float32)

        np.savez(out_path, **save_dict)