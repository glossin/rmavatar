import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.smplx_utils import smplx


def compute_vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    verts: (B, V, 3)
    faces: (F, 3)
    return: (B, V, 3)
    """
    tris = verts[:, faces.long()]  # (B, F, 3, 3)

    face_normals = torch.cross(
        tris[:, :, 1] - tris[:, :, 0],
        tris[:, :, 2] - tris[:, :, 0],
        dim=-1,
    )
    face_normals = F.normalize(face_normals, dim=-1, eps=1e-8)

    normals = torch.zeros_like(verts)
    for corner in range(3):
        index = faces[:, corner].long()
        normals.index_add_(1, index, face_normals)

    normals = F.normalize(normals, dim=-1, eps=1e-8)
    return normals


class PoseRefinementModule(nn.Module):
    """
    Anim-NeRF-style per-frame SMPL pose refinement for RMAvatar.

    It stores initial SMPL params as buffers and optimizes small deltas.
    """

    def __init__(
        self,
        smpl_params,
        smpl_config,
        optimize_body_pose=True,
        optimize_global_orient=True,
        optimize_transl=True,
        optimize_betas=False,
        device="cuda",
    ):
        super().__init__()

        self.device = torch.device(device)
        self.smpl_model = smplx.SMPL(**smpl_config).to(self.device)


        faces = torch.tensor(self.smpl_model.faces.astype(np.int64), device=self.device)
        self.register_buffer("faces", faces)

        body_pose = smpl_params["body_pose"].detach().float().to(self.device)
        global_orient = smpl_params["global_orient"].detach().float().to(self.device)
        transl = smpl_params["transl"].detach().float().to(self.device)
        betas = smpl_params["betas"].detach().float().to(self.device)

        if betas.ndim == 1:
            betas = betas[None, :]

        self.num_frames = body_pose.shape[0]

        self.register_buffer("body_pose_init", body_pose)
        self.register_buffer("global_orient_init", global_orient)
        self.register_buffer("transl_init", transl)
        self.register_buffer("betas_init", betas)

        self.delta_body_pose = nn.Parameter(
            torch.zeros_like(body_pose),
            requires_grad=optimize_body_pose,
        )
        self.delta_global_orient = nn.Parameter(
            torch.zeros_like(global_orient),
            requires_grad=optimize_global_orient,
        )
        self.delta_transl = nn.Parameter(
            torch.zeros_like(transl),
            requires_grad=optimize_transl,
        )
        self.delta_betas = nn.Parameter(
            torch.zeros_like(betas),
            requires_grad=optimize_betas,
        )

    def optimizable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _normalize_idx(self, idx):
        if not torch.is_tensor(idx):
            idx = torch.tensor([idx], dtype=torch.long, device=self.device)
        else:
            idx = idx.to(self.device).long().view(-1)
        return idx

    def get_refined_params(self, idx):
        idx = self._normalize_idx(idx)
        batch_size = idx.shape[0]

        body_pose = self.body_pose_init[idx] + self.delta_body_pose[idx]
        global_orient = self.global_orient_init[idx] + self.delta_global_orient[idx]
        transl = self.transl_init[idx] + self.delta_transl[idx]

        betas = self.betas_init + self.delta_betas
        if betas.shape[0] == 1 and batch_size > 1:
            betas = betas.expand(batch_size, -1).contiguous()

        return {
            "betas": betas,
            "body_pose": body_pose,
            "global_orient": global_orient,
            "transl": transl,
        }

    def forward(self, idx):
        idx = self._normalize_idx(idx)
        params = self.get_refined_params(idx)

        out = self.smpl_model(**params)
        verts = out["vertices"]  # (B, V, 3)
        normals = compute_vertex_normals(verts, self.faces)

        if verts.shape[0] != 1:
            raise NotImplementedError(
                "RMAvatar 当前训练循环默认每次处理一个 camera/frame。"
                "先保持 batch_size=1，后续再扩展多 batch。"
            )

        return {
            "mesh_verts": verts[0],
            "mesh_norms": normals[0],
            "mesh_faces": self.faces,
            "pose": params["body_pose"][0],
            "smpl_params": {k: v[0] if v.shape[0] == 1 else v for k, v in params.items()},
        }

    def regularization(self, idx, cfg):
        idx = self._normalize_idx(idx)

        losses = {}

        lambda_body = float(cfg.get("lambda_body_pose", 0.0))
        lambda_global = float(cfg.get("lambda_global_orient", 0.0))
        lambda_transl = float(cfg.get("lambda_transl", 0.0))

        if lambda_body > 0:
            losses["pose_body_prior"] = (
                self.delta_body_pose[idx].pow(2).mean() * lambda_body
            )

        if lambda_global > 0:
            losses["pose_global_prior"] = (
                self.delta_global_orient[idx].pow(2).mean() * lambda_global
            )

        if lambda_transl > 0:
            losses["pose_transl_prior"] = (
                self.delta_transl[idx].pow(2).mean() * lambda_transl
            )

        losses.update(self._smoothness_loss(idx, cfg))
        return losses

    def _smoothness_loss(self, idx, cfg):
        losses = {}

        lambda_body = float(cfg.get("lambda_smooth_body_pose", 0.0))
        lambda_global = float(cfg.get("lambda_smooth_global_orient", 0.0))
        lambda_transl = float(cfg.get("lambda_smooth_transl", 0.0))

        if lambda_body == 0 and lambda_global == 0 and lambda_transl == 0:
            return losses

        idx = idx.view(-1)
        valid_prev = idx > 0
        valid_next = idx < self.num_frames - 1

        neighbor_pairs = []
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

        if lambda_body > 0:
            losses["pose_body_smooth"] = smooth_delta(self.delta_body_pose) * lambda_body

        if lambda_global > 0:
            losses["pose_global_smooth"] = smooth_delta(self.delta_global_orient) * lambda_global

        if lambda_transl > 0:
            losses["pose_transl_smooth"] = smooth_delta(self.delta_transl) * lambda_transl

        return losses

    @torch.no_grad()
    def export_npz(self, out_path):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        body_pose = (self.body_pose_init + self.delta_body_pose).detach().cpu().numpy()
        global_orient = (
            self.global_orient_init + self.delta_global_orient
        ).detach().cpu().numpy()
        transl = (self.transl_init + self.delta_transl).detach().cpu().numpy()
        betas = (self.betas_init + self.delta_betas).detach().cpu().numpy()

        np.savez(
            out_path,
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
        )