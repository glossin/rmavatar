import os
from pathlib import Path
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import lpips
from tqdm import tqdm
from omegaconf import OmegaConf

from model import libcore
from model.rmavatar_model import SplattingAvatarModel
from model.loss_base import LossBase
from dataset.dataset_helper import make_frameset_data, make_dataloader
from model.pose_refinement import PoseRefinementModule


def to_scalar_idx(idx):
    if torch.is_tensor(idx):
        return int(idx.detach().cpu().view(-1)[0].item())
    return int(idx)


def freeze_model(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def build_pose_optimizer(pose_refiner, opt_cfg):
    param_groups = []

    if cfg_bool(opt_cfg, "optimize_body_pose", False):
        param_groups.append({
            "params": [pose_refiner.delta_body_pose],
            "lr": float(opt_cfg.get("lr_body_pose", 1.0e-6)),
        })

    if cfg_bool(opt_cfg, "optimize_global_orient", True):
        param_groups.append({
            "params": [pose_refiner.delta_global_orient],
            "lr": float(opt_cfg.get("lr_global_orient", 2.0e-6)),
        })

    if cfg_bool(opt_cfg, "optimize_transl", True):
        param_groups.append({
            "params": [pose_refiner.delta_transl],
            "lr": float(opt_cfg.get("lr_transl", 1.0e-5)),
        })

    if cfg_bool(opt_cfg, "optimize_betas", False):
        param_groups.append({
            "params": [pose_refiner.delta_betas],
            "lr": float(opt_cfg.get("lr_betas", 1.0e-7)),
        })

    if len(param_groups) == 0:
        raise ValueError(
            "No pose parameters enabled. "
            "Set at least one of optimize_body_pose / optimize_global_orient / "
            "optimize_transl / optimize_betas to true."
        )

    return torch.optim.Adam(
        param_groups,
        eps=float(opt_cfg.get("adam_eps", 1.0e-15)),
    )


def make_pose_reg_cfg(args):
    return {
        "lambda_body_pose": args.lambda_body_pose,
        "lambda_global_orient": args.lambda_global_orient,
        "lambda_transl": args.lambda_transl,
        "lambda_smooth_body_pose": args.lambda_smooth_body_pose,
        "lambda_smooth_global_orient": args.lambda_smooth_global_orient,
        "lambda_smooth_transl": args.lambda_smooth_transl,
    }

def cfg_bool(cfg, key, default=False):
    value = cfg.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in ["true", "1", "yes", "y"]
    return bool(value)


def resolve_path(dat_dir, path):
    if path is None:
        return None
    path = str(path)
    if os.path.isabs(path):
        return path
    return os.path.join(dat_dir, path)


if __name__ == "__main__":
    parser = ArgumentParser("RMAvatar pose-only refinement")

    parser.add_argument("--dat_dir", type=str, required=True)
    parser.add_argument(
        "--configs",
        type=lambda s: [i for i in s.split(";")],
        required=True,
        help="config files, e.g. configs/instant_avatar.yaml;configs/refine_pose_rmavatar.yaml",
    )


    args, extras = parser.parse_known_args()
    config = libcore.load_from_config(args.configs, cli_args=extras)

    config.dataset.dat_dir = args.dat_dir
    config.cache_dir = os.path.join(
        args.dat_dir,
        f"cache_{Path(args.configs[0]).stem}",
    )

    pose_cfg = config.get("pose_refine", None)
    if pose_cfg is None:
        raise KeyError(
            "Missing top-level `pose_refine:` section. "
            "Please add configs/refine_pose_rmavatar.yaml to --configs."
        )

    split = str(pose_cfg.get("split", "train"))
    ckpt_iteration = int(pose_cfg.get("iteration", config.optim.get("total_iteration", 60000)))
    deform_on = cfg_bool(pose_cfg, "deform_on", False)
    white_background = cfg_bool(
        pose_cfg,
        "white_background",
        config.optim.get("white_background", True),
    )
    refine_iters = int(pose_cfg.get("iters", 3000))
    save_every = int(pose_cfg.get("save_every", 500))
    grad_clip = float(pose_cfg.get("grad_clip", 1.0))
    opt_cfg = pose_cfg.get("optimizer", {})
    loss_cfg = pose_cfg.get("loss", {})
    pc_dir = pose_cfg.get("pc_dir", None)

    if pc_dir is None:
        model_path = str(
            pose_cfg.get("model_path", "output-splatting/baseline+offset(5w)")
        )
        model_path = resolve_path(args.dat_dir, model_path)
        pc_dir = os.path.join(
            model_path,
            "point_cloud",
            f"iteration_{ckpt_iteration}",
        )
    else:
        pc_dir = resolve_path(args.dat_dir, pc_dir)

    print(f"[PoseRefine] split: {split}")
    print(f"[PoseRefine] checkpoint iteration: {ckpt_iteration}")
    print(f"[PoseRefine] pc_dir: {pc_dir}")
    print(f"[PoseRefine] deform_on: {deform_on}")

    # args.optimize_body_pose = bool(args.optimize_body_pose)
    # args.optimize_global_orient = bool(args.optimize_global_orient)
    # args.optimize_transl = bool(args.optimize_transl)
    # args.optimize_betas = bool(args.optimize_betas)


    # config / dataset
    config.cache_dir = os.path.join(args.dat_dir, f"cache_{Path(args.configs[0]).stem}")

    frameset_train = make_frameset_data(config.dataset, split="train")
    frameset_target = make_frameset_data(config.dataset, split=split)
    dataloader = make_dataloader(frameset_target, shuffle=True)

    # canonical mesh must match training
    first_batch = frameset_train.__getitem__(0)
    cano_mesh = first_batch["mesh_info"]

    # load trained RMAvatar
    pipe = config.pipe
    gs_model = SplattingAvatarModel(config.model, cano_mesh, args, verbose=True).cuda()

    ply_fn = os.path.join(pc_dir, "point_cloud.ply")
    embed_fn = os.path.join(pc_dir, "embedding.json")

    if not os.path.exists(ply_fn):
        raise FileNotFoundError(ply_fn)

    if not os.path.exists(embed_fn):
        raise FileNotFoundError(embed_fn)

    gs_model.load_ply(ply_fn)
    gs_model.load_from_embedding(embed_fn)

    deform_fn = os.path.join(pc_dir, "deform.pth")

    if os.path.exists(deform_fn):
        gs_model.load_deform_weights(pc_dir, iteration=ckpt_iteration)
    elif deform_on:
        raise FileNotFoundError(
            f"deform_on=True but deform.pth does not exist in {pc_dir}"
        )

    freeze_model(gs_model)

    # pose refiner for target split
    pose_refiner = PoseRefinementModule(
        smpl_params=frameset_target.smpl_params,
        smpl_config=frameset_target.smpl_config,
        optimize_body_pose=cfg_bool(opt_cfg, "optimize_body_pose", False),
        optimize_global_orient=cfg_bool(opt_cfg, "optimize_global_orient", True),
        optimize_transl=cfg_bool(opt_cfg, "optimize_transl", True),
        optimize_betas=cfg_bool(opt_cfg, "optimize_betas", False),
        device="cuda",
    ).cuda()

    pose_optimizer = build_pose_optimizer(pose_refiner, opt_cfg)
    pose_reg_cfg = loss_cfg

    loss_base = LossBase(gs_model, config.optim).cuda()
    loss_fn_vgg = lpips.LPIPS(net="alex").cuda()

    out_pose = pose_cfg.get("out_pose", None)

    if out_pose is None:
        out_pose = os.path.join(
            args.dat_dir,
            "poses",
            f"anim_nerf_{split}_rmavatar_refined.npz",
        )
    else:
        out_pose = resolve_path(args.dat_dir, out_pose)

    os.makedirs(os.path.dirname(out_pose), exist_ok=True)

    print(f"[PoseRefine] out_pose: {out_pose}")

    os.makedirs(os.path.dirname(out_pose), exist_ok=True)

    data_iterator = iter(dataloader)
    camera_cache = {}

    pbar = tqdm(range(1, refine_iters + 1), desc=f"Pose refinement [{split}]")

    for refine_iter in pbar:
        try:
            batches = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            batches = next(data_iterator)

        batch = batches[0]
        idx = batch["idx"]

        mesh_info = pose_refiner(idx)
        pose = mesh_info["pose"]

        gs_model.update_to_cano_mesh(mesh_info)

        cache_key = to_scalar_idx(idx)
        if cache_key not in camera_cache:
            camera_cache[cache_key] = batch["scene_cameras"][0].cuda()

        viewpoint_cam = camera_cache[cache_key]
        viewpoint_cam.time = 0.0

        render_pkg = gs_model.render_to_camera(
            viewpoint_cam,
            pose,
            pipe,
            ckpt_iteration,
            ckpt_iteration,
            deform_on,
            white_background,
            itr=ckpt_iteration,
        )

        image = render_pkg["render"]
        gt_image = render_pkg["gt_image"]
        gt_alpha_mask = render_pkg.get("gt_alpha_mask", None)
        visibility_filter = render_pkg["visibility_filter"]
        offset = render_pkg.get("offset", None)

        loss = loss_base.collect_loss(
            image,
            gt_image,
            loss_fn_vgg,
            visibility_filter,
            viewpoint_cam,
            tb_writer=None,
            iteration=refine_iter,
            offset=offset,
            gt_alpha_mask=gt_alpha_mask,
        )

        pose_losses = pose_refiner.regularization(idx, pose_reg_cfg)
        for k, v in pose_losses.items():
            loss[k] = v
        if len(pose_losses) > 0:
            loss["total"] = loss["total"] + sum(pose_losses.values())

        pose_optimizer.zero_grad(set_to_none=True)
        loss["total"].backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                pose_refiner.optimizable_parameters(),
                grad_clip,
            )

        pose_optimizer.step()

        pbar.set_postfix({
            "loss": f"{loss['total'].item():.5f}",
        })

        if refine_iter % save_every == 0:
            pose_refiner.export_npz(out_pose)
            torch.save(
                {
                    "refine_iter": refine_iter,
                    "pose_refiner": pose_refiner.state_dict(),
                    "pose_optimizer": pose_optimizer.state_dict(),
                    "split": split,
                },
                out_pose.replace(".npz", ".pth"),
            )

    pose_refiner.export_npz(out_pose)
    torch.save(
        {
            "refine_iter": refine_iters,
            "pose_refiner": pose_refiner.state_dict(),
            "pose_optimizer": pose_optimizer.state_dict(),
            "split": split,
        },
        out_pose.replace(".npz", ".pth"),
    )

    print(f"[done] refined pose saved to: {out_pose}")