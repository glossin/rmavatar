from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import pytorch3d.structures.meshes as py3d_meshes

from scene.dataset_readers import convert_to_scene_cameras
from model import libcore


def _as_path(x) -> Path:
    return Path(os.path.expanduser(str(x))).resolve()


def _load_image_rgba(image_path: Path, mask_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8) * 255
    return np.concatenate([image, mask[..., None]], axis=-1)


def _make_camera_from_npz(cameras_npz: Path):
    data = np.load(cameras_npz)
    K = data["intrinsic"].astype(np.float32)
    extrinsic = data["extrinsic"].astype(np.float32)
    height = int(data["height"])
    width = int(data["width"])

    # RMAvatar/InstantAvatar reader convention: extrinsic is w2c.
    w2c = extrinsic
    R = w2c[:3, :3]
    T = w2c[:3, 3]

    cam = libcore.Camera()
    cam.h, cam.w = height, width
    cam.set_K(K)
    cam.R = R
    cam.setTranslation(T)
    return cam


def _mhr_to_rmavatar_space(vertices: torch.Tensor, use_mhr_coord_fix: bool) -> torch.Tensor:
    """
    Optional coordinate fix used when MHR TorchScript outputs vertices in cm and Y-up.
    Convert approximately to meter-scale camera-style coordinates: x, -y, -z.
    Keep this switch configurable because your local MHR/SAM export may already be aligned.
    """
    if not use_mhr_coord_fix:
        return vertices
    v = vertices.clone()
    v[..., 0] = v[..., 0] * 0.01
    v[..., 1] = -v[..., 1] * 0.01
    v[..., 2] = -v[..., 2] * 0.01
    return v


def _resolve_mhr_model_file(mhr_model_path) -> Path:
    """
    # [MHR-FACES 修改]
    兼容两种配置写法：
      1) mhr_model_path: /path/to/mhr_model.pt
      2) mhr_model_path: /path/to/mhr_assets_dir
    better_rigs_not_bigger_networks 的 MHRMesh 接收的是目录，然后内部拼接 mhr_model.pt；
    这里为了方便 RMAvatar 接入，同时支持文件路径和目录路径。
    """
    path = _as_path(mhr_model_path)
    if path.is_dir():
        path = path / "mhr_model.pt"
    if not path.exists():
        raise FileNotFoundError(f"MHR TorchScript model not found: {path}")
    return path


def _load_faces_from_mhr_torchscript(mhr_model: torch.jit.ScriptModule):
    """
    # [MHR-FACES 修改]
    参考 better_rigs_not_bigger_networks 的实现方式：
        ct = self._model.character_torch
        self._faces = ct.mesh.faces
        self._rest_vertices = ct.mesh.rest_vertices

    也就是说，faces 不再从外部 mhr_faces_lod1.npy 读取，而是直接从
    mhr_model.pt 内部的 character_torch.mesh.faces 读取。这样能保证 faces 和
    MHR forward 输出的 vertices 属于同一个 LOD / 同一个拓扑。
    """
    try:
        ct = mhr_model.character_torch
        faces = ct.mesh.faces.long().detach().cpu()
        rest_vertices = ct.mesh.rest_vertices.float().detach().cpu()
    except Exception as e:
        raise RuntimeError(
            "Failed to read faces from mhr_model.character_torch.mesh.faces. "
            "Please check that your mhr_model.pt is the SAM-3D-Body/MHR TorchScript asset "
            "that exposes character_torch.mesh.faces and character_torch.mesh.rest_vertices."
        ) from e

    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"Invalid MHR faces shape: {tuple(faces.shape)}; expected [F, 3].")
    if rest_vertices.ndim != 2 or rest_vertices.shape[-1] != 3:
        raise ValueError(
            f"Invalid MHR rest_vertices shape: {tuple(rest_vertices.shape)}; expected [N, 3]."
        )
    if faces.numel() > 0 and faces.max().item() >= rest_vertices.shape[0]:
        raise ValueError(
            "MHR faces are not compatible with rest_vertices: "
            f"faces.max={faces.max().item()}, num_vertices={rest_vertices.shape[0]}"
        )

    return faces, rest_vertices


class MHRInstantAvatarDataset(torch.utils.data.Dataset):
    """
    RMAvatar-compatible dataset reader for the output produced by
    convert_peoplesnapshot_to_rmavatar_mhr.py.

    Required layout:
      subject/
        config.json
        cameras.npz
        images/image_0000.png
        masks/mask_0000.png
        mhr/model_params.npz
        mhr/shape.npy

    Required config fields:
      dataset.frameset_type: mhr_instant_avatar
      dataset.dat_dir: /path/to/subject
      dataset.mhr_model_path: /path/to/mhr_model.pt
        or
      dataset.mhr_model_path: /path/to/mhr_assets_dir
      dataset.use_mhr_coord_fix: true/false

    # [MHR-FACES 修改]
    已删除对 dataset.mhr_faces_path 的强依赖。
    faces 会直接从 mhr_model.pt 内部读取：
        mhr_model.character_torch.mesh.faces
    因此通常不再需要额外准备 mhr_faces_lod1.npy。
    """

    def __init__(self, config, split="train", frm_list=None):
        self.config = config
        self.split = split
        self.dat_dir = _as_path(config.dat_dir)
        self.cameras_extent = float(config.get("cameras_extent", 1.0))
        self.data_device = config.get("data_device", "cuda")
        self.use_mhr_coord_fix = bool(config.get("use_mhr_coord_fix", True))
        self.precompute_vertices = bool(config.get("precompute_vertices", True))

        # [MHR-FACES 修改]
        # 统一管理 device。原文件多处直接 .cuda()，这里改成 self.mhr_device，
        # 方便后续在无 CUDA 或调试 CPU 时定位问题。
        self.mhr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.mhr_device.type != "cuda":
            print("[WARN] CUDA is not available. MHR forward will run on CPU and may be very slow.")

        self._load_frame_split(frm_list)
        self.cam = _make_camera_from_npz(self.dat_dir / "cameras.npz")
        self._load_mhr_assets_and_params()

        print(f"[MHRInstantAvatarDataset][{self.split}] num_frames = {len(self.frm_list)}")

    def _load_frame_split(self, frm_list):
        if frm_list is not None:
            self.frm_list = list(frm_list)
            return

        with open(self.dat_dir / "config.json", "r", encoding="utf-8") as f:
            contents = json.load(f)
        s = contents[self.split]
        self.frm_list = list(range(int(s["start"]), int(s["end"]) + 1, int(s.get("skip", 1))))

    def _load_mhr_assets_and_params(self):
        mhr_npz = np.load(self.dat_dir / "mhr" / "model_params.npz")
        self.model_parameters = torch.from_numpy(mhr_npz["model_parameters"].astype(np.float32))
        self.identity_coeffs = torch.from_numpy(mhr_npz["identity_coeffs"].astype(np.float32))
        if "expression_coeffs" in mhr_npz:
            self.expression_coeffs = torch.from_numpy(mhr_npz["expression_coeffs"].astype(np.float32))
        else:
            self.expression_coeffs = torch.zeros((self.model_parameters.shape[0], 72), dtype=torch.float32)

        self.pred_cam_t = None
        if "pred_cam_t" in mhr_npz:
            self.pred_cam_t = torch.from_numpy(mhr_npz["pred_cam_t"].astype(np.float32))

        # [MHR-FACES 修改]
        # 原代码：
        #   mhr_model_path = _as_path(self.config.mhr_model_path)
        #   faces_path = _as_path(self.config.mhr_faces_path)
        #   self.mhr_model = torch.jit.load(str(mhr_model_path)).cuda().eval()
        #   self.mhr_faces = torch.from_numpy(np.load(faces_path).astype(np.int64))
        #
        # 现在：只需要 mhr_model_path，不需要 mhr_faces_path。
        # faces 直接从 TorchScript 模型内部 character_torch.mesh.faces 获得。
        mhr_model_file = _resolve_mhr_model_file(self.config.mhr_model_path)
        self.mhr_model = torch.jit.load(str(mhr_model_file), map_location=self.mhr_device)
        self.mhr_model.eval()

        # [MHR-FACES 修改]
        # 参考 better_rigs_not_bigger_networks/src/gaussian_avatar/models/mesh/mhr.py：
        #   ct = self._model.character_torch
        #   self._rest_vertices = ct.mesh.rest_vertices
        #   self._faces = ct.mesh.faces
        self.mhr_faces, self.mhr_rest_vertices = _load_faces_from_mhr_torchscript(self.mhr_model)

        # print("[MHR] model file:", mhr_model_file)
        # print("[MHR] rest_vertices:", tuple(self.mhr_rest_vertices.shape))
        # print("[MHR] faces:", tuple(self.mhr_faces.shape))
        # print(
        #     "[MHR] faces min/max:",
        #     int(self.mhr_faces.min().item()),
        #     int(self.mhr_faces.max().item()),
        # )

        self.mhr_verts = None
        if self.precompute_vertices:
            verts = []
            with torch.no_grad():
                for i in range(self.model_parameters.shape[0]):
                    v = self._forward_mhr_vertices(i)
                    verts.append(v.cpu())
            self.mhr_verts = torch.stack(verts, dim=0)

        init_verts = self._get_vertices_cpu(0)[None]
        self.mesh_py3d = py3d_meshes.Meshes(init_verts, self.mhr_faces[None])

        # if self.pred_cam_t is not None:
        #     print("[MHR] pred_cam_t shape:", tuple(self.pred_cam_t.shape))
        #     print("[MHR] pred_cam_t first:", self.pred_cam_t[0])
        #     print("[MHR] pred_cam_t min:", self.pred_cam_t.min(dim=0).values)
        #     print("[MHR] pred_cam_t max:", self.pred_cam_t.max(dim=0).values)

    def _forward_mhr_vertices(self, frame_idx: int) -> torch.Tensor:
        identity = self.identity_coeffs[None].to(self.mhr_device)
        model_params = self.model_parameters[frame_idx:frame_idx + 1].to(self.mhr_device)
        expr = self.expression_coeffs[frame_idx:frame_idx + 1].to(self.mhr_device)

        # [MHR-FACES 修改]
        # 这里仍然使用 TorchScript MHR 前向：
        #   vertices, skeleton_state = self.mhr_model(identity, model_params, expr)
        # faces 的读取方式改变，不影响这里的 forward 逻辑。
        vertices, _ = self.mhr_model(identity, model_params, expr)

        vertices = _mhr_to_rmavatar_space(vertices[0], self.use_mhr_coord_fix)

        # 关键修复：把全局相机平移真正加到 mesh 顶点上
        # 如果 pred_cam_t 是 SAM/HMR 常见的 camera translation，通常应该是米制相机坐标，
        # 因此不要再乘 0.01，也不要再翻 y/z。
        if self.pred_cam_t is not None:
            t = self.pred_cam_t[frame_idx].to(vertices.device)
            vertices = vertices + t[None, :]

        return vertices

    def _get_vertices_cpu(self, frame_idx: int) -> torch.Tensor:
        if self.mhr_verts is not None:
            return self.mhr_verts[frame_idx]
        with torch.no_grad():
            return self._forward_mhr_vertices(frame_idx).cpu()

    def __len__(self):
        return len(self.frm_list)

    def __getitem__(self, idx):
        frm_idx = self.frm_list[idx]
        image_path = self.dat_dir / "images" / f"image_{frm_idx:04d}.png"
        mask_path = self.dat_dir / "masks" / f"mask_{frm_idx:04d}.png"
        rgba = _load_image_rgba(image_path, mask_path)

        color_frames = libcore.DataVec()
        color_frames.cams = [self.cam]
        color_frames.frames = [rgba]
        color_frames.images_path = [str(image_path)]
        scene_cameras = convert_to_scene_cameras(color_frames, self.config)

        batch = {
            "idx": idx,
            "frm_idx": frm_idx,
            "color_frames": color_frames,
            "scene_cameras": scene_cameras,
            "cameras_extent": self.cameras_extent,
            "mesh_info": self.get_mhr_mesh(frm_idx),
        }
        return batch

    def get_mhr_mesh(self, frm_idx: int):
        verts = self._get_vertices_cpu(frm_idx)

        # [MHR-FACES 修改]
        # 原代码写法：
        #   frame_mesh = self.mesh_py3d.update_padded(self.mhr_verts[frm_idx:frm_idx + 1])
        # 这个写法在 precompute_vertices=False 时会因为 self.mhr_verts is None 报错。
        # 改成统一使用上面 _get_vertices_cpu() 得到的 verts，兼容预计算和动态 forward 两种模式。
        frame_mesh = self.mesh_py3d.update_padded(verts[None])

        device = torch.device("cuda" if self.data_device == "cuda" and torch.cuda.is_available() else "cpu")
        mesh_info = {
            "mesh_verts": frame_mesh.verts_packed().to(device),
            "mesh_norms": frame_mesh.verts_normals_packed().to(device),
            "mesh_faces": frame_mesh.faces_packed().to(device),
            "pose": self.model_parameters[frm_idx].to(device),
            "mhr_identity": self.identity_coeffs.to(device),
            "mhr_expr": self.expression_coeffs[frm_idx].to(device),
        }
        if self.pred_cam_t is not None:
            mesh_info["pred_cam_t"] = self.pred_cam_t[frm_idx].to(device)
        return mesh_info
