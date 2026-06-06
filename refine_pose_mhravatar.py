"""
python refine_pose_mhravatar.py \
  --dat_dir /root/data1/project/rmavatar-mhr/datasets/PeopleSnapshot/111/male-3-casual \
  --configs "configs/peoplesnapshot.yaml;configs/mhr_avatar.yaml;configs/refine_pose_mhravatar.yaml"
"""

import os
import math
import json
from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
import lpips
from tqdm import tqdm
from omegaconf import OmegaConf

from model import libcore
from model.rmavatar_model import SplattingAvatarModel
from model.loss_base import LossBase
from dataset.dataset_helper import make_frameset_data, make_dataloader
from model.mhr_pose_refinement import MHRPoseRefinementModule


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

    p = Path(os.path.expanduser(str(path)))

    if p.is_absolute():
        return str(p)

    if p.exists():
        return str(p.resolve())

    return str(Path(dat_dir) / p)


def freeze_model(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def build_pose_optimizer(refiner, opt_cfg):
    param_groups = []

    if cfg_bool(opt_cfg, "optimize_model_parameters", False):
        param_groups.append({
            "params": [refiner.delta_model_parameters],
            "lr": float(opt_cfg.get("lr_model_parameters", 1.0e-6)),
        })

    if cfg_bool(opt_cfg, "optimize_expression", False):
        param_groups.append({
            "params": [refiner.delta_expression],
            "lr": float(opt_cfg.get("lr_expression", 1.0e-6)),
        })

    if cfg_bool(opt_cfg, "optimize_pred_cam_t", True) or cfg_bool(opt_cfg, "optimize_extra_transl", True):
        param_groups.append({
            "params": [refiner.delta_pred_cam_t],
            "lr": float(opt_cfg.get("lr_pred_cam_t", 5.0e-6)),
        })

    if len(param_groups) == 0:
        raise ValueError(
            "No MHR pose parameters enabled. "
            "Enable optimize_pred_cam_t or optimize_model_parameters."
        )

    return torch.optim.Adam(
        param_groups,
        eps=float(opt_cfg.get("adam_eps", 1.0e-15)),
    )


def foreground_l1_loss(image, gt_image, gt_alpha_mask, eps=1.0e-6):
    if gt_alpha_mask is None:
        return None

    mask = gt_alpha_mask.float().to(image.device)

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    if mask.dim() == 3 and mask.shape[0] != 1:
        mask = mask[:1]

    return ((image - gt_image).abs() * mask).sum() / (
        mask.sum() * image.shape[0] + eps
    )


if __name__ == "__main__":
    parser = ArgumentParser("mhravatar pose-only refinement")

    parser.add_argument("--dat_dir", type=str, required=True)
    parser.add_argument(
        "--configs",
        type=lambda s: [i for i in s.split(";")],
        required=True,
    )
    
    args, extras = parser.parse_known_args()

    config = libcore.load_from_config(args.configs, cli_args=extras)
    config.dataset.dat_dir = args.dat_dir
    config.cache_dir = os.path.join(
        args.dat_dir,
        f"cache_{Path(args.configs[0]).stem}",
    )

    refine_cfg = config.get("mhr_pose_refine", None)

    if refine_cfg is None:
        raise KeyError(
            "Missing top-level `mhr_pose_refine:` config. "
            "Please add configs/refine_pose_mhravatar.yaml."
        )

    split = str(refine_cfg.get("split", "test"))
    iteration = int(refine_cfg.get("iteration", config.optim.get("total_iteration", 50000)))
    total_iteration = int(refine_cfg.get("total_iteration", iteration))

    deform_on = cfg_bool(refine_cfg, "deform_on", False)
    white_background = cfg_bool(refine_cfg, "white_background", True)

    refine_iters = int(refine_cfg.get("iters", 3000))
    save_every = int(refine_cfg.get("save_every", 500))
    grad_clip = float(refine_cfg.get("grad_clip", 1.0))

    opt_cfg = refine_cfg.get("optimizer", {})
    loss_cfg = refine_cfg.get("loss", {})

    pc_dir = refine_cfg.get("pc_dir", None)
    if pc_dir is None:
        model_path = resolve_path(
            args.dat_dir,
            refine_cfg.get("model_path", "output-splatting/baseline+offset(5w)"),
        )
        pc_dir = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    else:
        pc_dir = resolve_path(args.dat_dir, pc_dir)

    out_params = refine_cfg.get("out_params", None)
    if out_params is None:
        out_params = os.path.join(
            args.dat_dir,
            "mhr",
            f"model_params_refined_{split}.npz",
        )
    else:
        out_params = resolve_path(args.dat_dir, out_params)

    if not out_params.endswith(".npz"):
        raise ValueError(f"out_params must be a .npz file path, got: {out_params}")

    os.makedirs(os.path.dirname(out_params), exist_ok=True)

    print(f"[MHRPoseRefine] dat_dir: {args.dat_dir}")
    print(f"[MHRPoseRefine] split: {split}")
    print(f"[MHRPoseRefine] iteration: {iteration}")
    print(f"[MHRPoseRefine] pc_dir: {pc_dir}")
    print(f"[MHRPoseRefine] out_params: {out_params}")
    print(f"[MHRPoseRefine] deform_on: {deform_on}")

    # Dataset.
    frameset_train = make_frameset_data(config.dataset, split="train")
    frameset_target = make_frameset_data(config.dataset, split=split)
    dataloader = make_dataloader(frameset_target, shuffle=True)

    # Canonical mesh must match training.
    first_batch = frameset_train.__getitem__(0)
    cano_mesh = first_batch["mesh_info"]

    pipe = config.pipe

    model_args = Namespace(
        dat_dir=args.dat_dir,
        deform_on=int(deform_on),
    )

    gs_model = SplattingAvatarModel(
        config.model,
        cano_mesh,
        model_args,
        verbose=True,
    ).cuda()

    ply_fn = os.path.join(pc_dir, "point_cloud.ply")
    embed_fn = os.path.join(pc_dir, "embedding.json")
    deform_fn = os.path.join(pc_dir, "deform.pth")

    missing = []

    if not os.path.exists(ply_fn):
        missing.append(ply_fn)
    if not os.path.exists(embed_fn):
        missing.append(embed_fn)
    if deform_on and not os.path.exists(deform_fn):
        missing.append(deform_fn)

    if missing:
        raise FileNotFoundError(
            "[MHRPoseRefine] Invalid checkpoint directory.\n"
            f"pc_dir: {pc_dir}\n"
            "Missing:\n  " + "\n  ".join(missing)
        )

    gs_model.load_ply(ply_fn)
    gs_model.load_from_embedding(embed_fn)

    if os.path.exists(deform_fn):
        gs_model.load_deform_weights(pc_dir, iteration=iteration)

    freeze_model(gs_model)

    # MHR pose refiner.
    pose_refiner = MHRPoseRefinementModule(
        frameset=frameset_target,
        optimize_model_parameters=cfg_bool(opt_cfg, "optimize_model_parameters", False),
        optimize_expression=cfg_bool(opt_cfg, "optimize_expression", False),
        optimize_pred_cam_t=cfg_bool(opt_cfg, "optimize_pred_cam_t", True),
        optimize_extra_transl=cfg_bool(opt_cfg, "optimize_extra_transl", True),
        device="cuda",
    ).cuda()

    # Optional clamps.
    max_pred_cam_t_delta = float(opt_cfg.get("max_pred_cam_t_delta", 0.0))
    pose_refiner.max_pred_cam_t_delta = (
        max_pred_cam_t_delta if max_pred_cam_t_delta > 0 else None
    )

    max_model_delta = float(opt_cfg.get("max_model_delta", 0.0))
    pose_refiner.max_model_delta = (
        max_model_delta if max_model_delta > 0 else None
    )

    max_expression_delta = float(opt_cfg.get("max_expression_delta", 0.0))
    pose_refiner.max_expression_delta = (
        max_expression_delta if max_expression_delta > 0 else None
    )

    print(
        "[MHRPoseRefine] clamp: "
        f"max_pred_cam_t_delta={pose_refiner.max_pred_cam_t_delta}, "
        f"max_model_delta={pose_refiner.max_model_delta}, "
        f"max_expression_delta={pose_refiner.max_expression_delta}"
    )

    pose_optimizer = build_pose_optimizer(pose_refiner, opt_cfg)

    loss_base = LossBase(gs_model, config.optim).cuda()
    loss_fn_vgg = lpips.LPIPS(net="alex").cuda()

    data_iterator = iter(dataloader)
    camera_cache = {}

    pbar = tqdm(
        range(1, refine_iters + 1),
        desc=f"MHR pose refinement [{split}]",
    )

    for refine_iter in pbar:
        try:
            batches = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            batches = next(data_iterator)

        batch = batches[0]
        idx = batch["idx"]

        mesh_info = pose_refiner(idx)
        pose = mesh_info.get("pose", None)

        gs_model.update_to_cano_mesh(mesh_info)

        cache_key = int(idx.detach().cpu().view(-1)[0].item()) if torch.is_tensor(idx) else int(idx)

        if cache_key not in camera_cache:
            camera_cache[cache_key] = batch["scene_cameras"][0].cuda()

        viewpoint_cam = camera_cache[cache_key]
        viewpoint_cam.time = 0.0

        render_pkg = gs_model.render_to_camera(
            viewpoint_cam,
            pose,
            pipe,
            iteration,
            total_iteration,
            deform_on,
            white_background,
            itr=iteration,
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

        pose_losses = pose_refiner.regularization(idx, loss_cfg)

        for k, v in pose_losses.items():
            loss[k] = v

        if pose_losses:
            loss["total"] = loss["total"] + sum(pose_losses.values())

        lambda_fg_l1 = float(loss_cfg.get("lambda_fg_l1", 0.0))

        if lambda_fg_l1 > 0:
            fg_l1 = foreground_l1_loss(image, gt_image, gt_alpha_mask)

            if fg_l1 is not None:
                loss["mhr_fg_l1"] = fg_l1 * lambda_fg_l1
                loss["total"] = loss["total"] + loss["mhr_fg_l1"]

        pose_optimizer.zero_grad(set_to_none=True)
        loss["total"].backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                pose_refiner.optimizable_parameters(),
                grad_clip,
            )

        pose_optimizer.step()

        pbar.set_postfix({"loss": f"{loss['total'].item():.5f}"})

        if save_every > 0 and refine_iter % save_every == 0:
            pose_refiner.export_full_npz(out_params)

            torch.save(
                {
                    "refine_iter": refine_iter,
                    "pose_refiner": pose_refiner.state_dict(),
                    "pose_optimizer": pose_optimizer.state_dict(),
                    "split": split,
                    "out_params": out_params,
                },
                str(Path(out_params).with_suffix(".pth")),
            )

    pose_refiner.export_full_npz(out_params)

    torch.save(
        {
            "refine_iter": refine_iters,
            "pose_refiner": pose_refiner.state_dict(),
            "pose_optimizer": pose_optimizer.state_dict(),
            "split": split,
            "out_params": out_params,
        },
        str(Path(out_params).with_suffix(".pth")),
    )

    config_save_path = Path(out_params).with_name(
        f"{Path(out_params).stem}_refine_config.yaml"
    )
    OmegaConf.save(config, str(config_save_path))

    print(f"[MHRPoseRefine] done")
    print(f"[MHRPoseRefine] saved params: {out_params}")
    print(f"[MHRPoseRefine] saved ckpt: {Path(out_params).with_suffix('.pth')}")
    print(f"[MHRPoseRefine] saved config: {config_save_path}")