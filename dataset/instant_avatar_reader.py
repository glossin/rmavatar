# InstantAvatar/PeopleSnapshot data Reader.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import os
import cv2
import json
from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
from scene.dataset_readers import convert_to_scene_cameras
from model import libcore
from model.smplx_utils import smplx
import pytorch3d.structures.meshes as py3d_meshes
from utils.network_util import initseq, RodriguesModule
from model.smplx_utils.smplx.lbs import batch_rodrigues



def read_instant_avatar_frameset(dat_dir, frm_idx, cam, extension='.png'):
    image_path = os.path.join(dat_dir, f'images/image_{frm_idx:04d}.png')
    mask_path = os.path.join(dat_dir, f'masks/mask_{frm_idx:04d}.png')
    
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    #mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)[:, :, 0]
    image = np.concatenate([image, mask[:, :, None]], axis=-1)

    color_frames = libcore.DataVec()
    color_frames.cams = [cam]
    color_frames.frames = [image]
    color_frames.images_path = [image_path]
    return color_frames

class InstantAvatarDataset(torch.utils.data.Dataset):
    def __init__(self, config, split='train', frm_list=None):
        self.config = config
        self.split = split

        self.dat_dir = config.dat_dir
        self.cameras_extent = config.get('cameras_extent', 1.0)

        self.rodriguez = RodriguesModule()
        self.block_mlps = nn.Sequential(
            nn.Linear(69, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 69)
        ).cuda()

        self.load_config_file()
        self.load_camera_file()
        self.load_pose_file()
        self.num_frames = len(self.frm_list)
        print(f'[InstantAvatarDataset][{self.split}] num_frames = {self.num_frames}')

        ##################################################
    # load config.json
    def load_config_file(self):
        if not os.path.exists(os.path.join(self.dat_dir, 'config.json')):
            raise NotImplementedError
        
        with open(os.path.join(self.dat_dir, 'config.json'), 'r') as fp:
            contents = json.load(fp)

        self.start_idx = contents[self.split]['start']
        self.end_idx = contents[self.split]['end']
        self.step = contents[self.split]['skip']
        self.frm_list = [i for i in range(self.start_idx, self.end_idx+1, self.step)]

        smpl_model_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '..', 'model', 'smplx_utils', 'smplx_models', 'smpl'
        ))
        self.smpl_config = {
            'model_path': smpl_model_path,
            'model_type': 'smpl',
            'gender': contents['gender'],
        }

    # load cameras.npz
    def load_camera_file(self):
        if not os.path.exists(os.path.join(self.dat_dir, 'cameras.npz')):
            raise NotImplementedError

        contents = np.load(os.path.join(self.dat_dir, 'cameras.npz'))
        K = contents["intrinsic"]
        c2w = np.linalg.inv(contents["extrinsic"])
        height = contents["height"]
        width = contents["width"]
        w2c = np.linalg.inv(c2w)

        R = w2c[:3,:3]
        T = w2c[:3, 3]

        cam = libcore.Camera()
        cam.h, cam.w = height, width
        cam.set_K(K)
        cam.R = R
        cam.setTranslation(T)
        # print(cam)
        self.cam = cam

    # load poses.npz
    def load_pose_file(self):
        if not os.path.exists(os.path.join(self.dat_dir, 'poses.npz')):
            raise NotImplementedError

        smpl_params = dict(np.load(os.path.join(self.dat_dir, 'poses.npz')))  # shape, pose, xyz

        if "thetas" in smpl_params:
            smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
            smpl_params["global_orient"] = smpl_params["thetas"][..., :3]

        smpl_device = torch.device("cpu")
        n_frames = len(self.frm_list)

        # 先读取原始 poses.npz
        self.smpl_params = {
            "betas": torch.tensor(
                smpl_params["betas"].astype(np.float32).reshape(1, 10)
            ).to(smpl_device),

            "body_pose": torch.tensor(
                smpl_params["body_pose"].astype(np.float32)
            )[self.frm_list].to(smpl_device),

            "global_orient": torch.tensor(
                smpl_params["global_orient"].astype(np.float32)
            )[self.frm_list].to(smpl_device),

            "transl": torch.tensor(
                smpl_params["transl"].astype(np.float32)
            )[self.frm_list].to(smpl_device),
        }

        # load refined
        refine_fn = os.path.join(self.dat_dir, f"poses/anim_nerf_{self.split}.npz")

        if os.path.exists(refine_fn):
            print(f'[InstantAvatar] use refined smpl: {refine_fn}')

            split_smpl_params = np.load(refine_fn)
            refined_keys = [k for k in split_smpl_params.files if k != 'betas']

            # -----------------------------
            # 关键修复 1：betas 必须是 [1, 10] 或 [N, 10]
            # -----------------------------
            betas = torch.tensor(split_smpl_params['betas']).float()

            if betas.ndim == 1:
                betas = betas.reshape(1, -1)

            # SMPL 这里只用前 10 个 shape coefficients
            betas = betas[:, :10]

            # 一般 betas 是全局 shape，所以保留 [1, 10] 即可
            # 如果 refined 文件里给了逐帧 betas，并且帧数正好匹配，也允许 [N, 10]
            if betas.shape[0] != 1 and betas.shape[0] != n_frames:
                print(
                    f"[WARN][{self.split}] refined betas shape = {betas.shape}, "
                    f"but n_frames = {n_frames}. Use first beta only."
                )
                betas = betas[:1]

            self.smpl_params['betas'] = betas.to(smpl_device)

            # -----------------------------
            # 关键修复 2：refined pose 参数要和当前 split 的帧数对齐
            # -----------------------------
            for key in refined_keys:
                value = split_smpl_params[key].astype(np.float32)

                # 如果 refined 文件里已经是 train/test split 后的长度，例如 298 或 350，直接用
                if value.ndim > 0 and value.shape[0] == n_frames:
                    value = value

                # 如果 refined 文件里是完整视频长度，则按 self.frm_list 取当前 split
                elif value.ndim > 0 and value.shape[0] > max(self.frm_list):
                    value = value[self.frm_list]

                # 其他情况先保留原值，但打印出来方便排查
                else:
                    print(
                        f"[WARN][{self.split}] refined key {key} has shape {value.shape}, "
                        f"n_frames = {n_frames}, max frm_list = {max(self.frm_list)}"
                    )

                self.smpl_params[key] = torch.tensor(value).float().to(smpl_device)

        # -----------------------------
        # 只在最终 smpl_params 确定后创建 SMPL model
        # -----------------------------
        self.smpl_model = smplx.SMPL(**self.smpl_config).to(smpl_device)

        with torch.no_grad():
            out = self.smpl_model(**self.smpl_params)

        self.smpl_verts = out['vertices'].detach().cpu()

        # -----------------------------
        # 关键修复 3：检查 self.smpl_verts 帧数
        # -----------------------------
        print(
            f"[DEBUG][{self.split}] "
            f"n_frames={n_frames}, "
            f"smpl_verts={self.smpl_verts.shape}, "
            f"betas={self.smpl_params['betas'].shape}, "
            f"body_pose={self.smpl_params['body_pose'].shape}, "
            f"global_orient={self.smpl_params['global_orient'].shape}, "
            f"transl={self.smpl_params['transl'].shape}"
        )

        assert self.smpl_verts.shape[0] == n_frames, (
            f"[{self.split}] SMPL verts frame mismatch: "
            f"self.smpl_verts.shape[0] = {self.smpl_verts.shape[0]}, "
            f"but n_frames = {n_frames}. "
            f"Please check {refine_fn}."
        )

        # -----------------------------
        # 关键修复 4：用最终的 self.smpl_verts 初始化 mesh_py3d
        # -----------------------------
        faces = torch.tensor(
            self.smpl_model.faces[None, ...].astype(np.int64),
            dtype=torch.long,
            device=self.smpl_verts.device,
        )

        self.mesh_py3d = py3d_meshes.Meshes(
            self.smpl_verts[:1],
            faces
        )
    ##################################################
    def __len__(self):
        return len(self.frm_list)

    def __getitem__(self, idx):
        if idx is None:
            idx = torch.randint(0, len(self.frm_list), (1,)).item()

        frm_idx = self.frm_list[idx]

        # frames
        color_frames = read_instant_avatar_frameset(self.dat_dir, frm_idx, self.cam)
        scene_cameras = convert_to_scene_cameras(color_frames, self.config)
        
        batch = {
            'idx': idx,
            'frm_idx': frm_idx,
            'color_frames': color_frames,
            'scene_cameras': scene_cameras,
            'cameras_extent': self.cameras_extent,
        }

        # mesh
        batch['mesh_info'] = self.get_smpl_mesh(idx)
        
        return batch

    def get_smpl_mesh(self, idx):
        verts = self.smpl_verts[idx:idx + 1]
        if verts.shape[0] != 1:
            raise IndexError(
                f"[{self.split}] invalid SMPL vertex slice: "
                f"idx={idx}, verts.shape={verts.shape}, "
                f"all_smpl_verts.shape={self.smpl_verts.shape}, "
                f"num_frames={len(self.frm_list)}"
        )
        frame_mesh = self.mesh_py3d.update_padded(verts)
        # a=frame_mesh.verts_packed()
        # b=frame_mesh.faces_packed()
        # save_obj('smpl392.obj', a, b)
        return {
            'mesh_verts': frame_mesh.verts_packed(),
            'mesh_norms': frame_mesh.verts_normals_packed(),
            'mesh_faces': frame_mesh.faces_packed(),
            'pose': self.smpl_params["body_pose"][idx],
        }

    # def save_obj(filename, verts, faces):
    #     with open(filename, 'w') as f:
    #         for vert in verts:
    #             f.write("v {} {} {}\n".format(vert[0], vert[1], vert[2]))  # 写入顶点坐标
    #         for face in faces:
    #             f.write("f")
    #             for idx in face:
    #                 f.write(" {}".format(idx + 1))  # 索引从1开始
    #             f.write("\n")



    

    