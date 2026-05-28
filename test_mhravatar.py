#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone evaluator for mhravatar / RMAvatar mhr-rmavatar branch.

Example:
  python test_mhravatar.py \
    --configs  "configs/peoplesnapshot.yaml;configs/mhr_avatar.yaml" \
    --dat_dir /root/data1/datasets/PeopleSnapshot_preprocessed/female-3-casual \
    --model_path /root/data1/project/rmavatar-mhr/out/female-3-casual-deform_on11 \ #checkpoint的路径 有iteration 50000 会加上//point_cloud/iteration_50000/point_cloud.ply
    --out_dir /root/data1/project/rmavatar-mhr/out/test \
    --iteration 50000 \ #不加iteration会默认找最新的
    --deform_on 1

If --configs is omitted, the script reads <model_path>/config.yaml and still
overrides config.dataset.dat_dir with --dat_dir, matching train_rmavatar.py.
"""

import json
import os
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from dataset.dataset_helper import make_frameset_data
from model import libcore
from model.loss_base import testing_routine
from model.rmavatar_model import SplattingAvatarModel


def _split_configs(configs: Optional[str]) -> list[str]:
    if configs is None or configs == "":
        return []
    return [p for p in configs.split(";") if p]


def _resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def _parse_iteration_from_dir(path: Path) -> Optional[int]:
    m = re.fullmatch(r"iteration_(\d+)", path.name)
    return int(m.group(1)) if m else None


def _find_latest_checkpoint(model_path: str) -> Tuple[int, Path]:
    pc_root = Path(model_path) / "point_cloud"
    candidates: list[Tuple[int, Path]] = []
    for p in pc_root.glob("iteration_*"):
        it = _parse_iteration_from_dir(p)
        if it is not None and (p / "point_cloud.ply").exists():
            candidates.append((it, p))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {pc_root}/iteration_*/point_cloud.ply")
    return max(candidates, key=lambda x: x[0])


def _resolve_checkpoint(model_path: str, pc_dir: Optional[str], iteration: int) -> Tuple[int, Path]:
    if pc_dir:
        p = Path(pc_dir)
        if not p.is_absolute():
            p = Path(model_path) / p
        p = p.resolve()
        if not (p / "point_cloud.ply").exists():
            raise FileNotFoundError(f"Missing point_cloud.ply in {p}")
        inferred = _parse_iteration_from_dir(p)
        if iteration < 0:
            if inferred is None:
                raise ValueError("--iteration is required when --pc_dir is not named iteration_<N>")
            iteration = inferred
        return iteration, p

    if iteration < 0:
        return _find_latest_checkpoint(model_path)

    p = Path(model_path) / "point_cloud" / f"iteration_{iteration}"
    if not (p / "point_cloud.ply").exists():
        raise FileNotFoundError(f"Missing checkpoint: {p / 'point_cloud.ply'}")
    return iteration, p


def _as_cuda_float(data, name: str) -> torch.Tensor:
    t = torch.as_tensor(data, dtype=torch.float32, device="cuda")
    if not torch.isfinite(t).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return t


def _restore_embedding_local_params(gs_model: SplattingAvatarModel, pc_dir: Path, allow_missing: bool) -> None:
    """
    The saved PLY contains full Gaussian attributes, but in the mesh-bound case
    point_cloud.ply stores get_xyz/get_rotation. embedding.json stores local
    _xyz/_rotation, so restore these two after load_ply.
    """
    embed_path = pc_dir / "embedding.json"
    if not embed_path.exists():
        if allow_missing:
            print(f"[WARN] Missing {embed_path}; using xyz/rotation from point_cloud.ply directly.")
            return
        raise FileNotFoundError(
            f"Missing {embed_path}. For mesh-bound mhravatar checkpoints, "
            "embedding.json is needed to restore local _xyz/_rotation correctly. "
            "Pass --allow_missing_embedding only if you know the PLY stores local values."
        )

    with open(embed_path, "r") as f:
        emb = json.load(f)

    for key in ("_xyz", "_rotation"):
        if key not in emb:
            raise KeyError(f"{embed_path} does not contain {key}")
        value = _as_cuda_float(emb[key], key)
        old = getattr(gs_model, key)
        if old.numel() > 0 and old.shape[0] != value.shape[0]:
            raise RuntimeError(f"{key} length mismatch: ply={old.shape}, embedding={value.shape}")
        setattr(gs_model, key, nn.Parameter(value.detach().clone(), requires_grad=False))

    print(f"[load] restored local _xyz/_rotation from {embed_path}")


def _load_checkpoint(
    gs_model: SplattingAvatarModel,
    pc_dir: Path,
    deform_on: bool,
    allow_missing_embedding: bool = False,
) -> None:
    ply_path = pc_dir / "point_cloud.ply"
    print(f"[load] gaussian ply: {ply_path}")
    # Use init_params=True because create_from_canonical() has already registered
    # _xyz/_features/_opacity/_scaling/_rotation as nn.Parameter objects.
    # Passing init_params=False would try to assign a plain Tensor over an
    # existing Parameter and causes:
    # TypeError: cannot assign torch.cuda.FloatTensor as parameter _xyz
    gs_model.load_ply(str(ply_path), init_params=True)

    # load_ply currently restores binding as int32; indexing is safer as int64.
    if getattr(gs_model, "binding", None) is not None:
        gs_model.binding = gs_model.binding.long()
        n_faces = int(gs_model.cano_faces.shape[0])
        gs_model.binding_counter = torch.zeros(n_faces, dtype=torch.int32, device="cuda")
        gs_model.binding_counter.scatter_add_(
            0,
            gs_model.binding,
            torch.ones_like(gs_model.binding, dtype=torch.int32, device="cuda"),
        )

    _restore_embedding_local_params(gs_model, pc_dir, allow_missing_embedding)

    deform_path = pc_dir / "deform.pth"
    if deform_on:
        if not deform_path.exists():
            raise FileNotFoundError(
                f"--deform_on 1 was requested, but deformation weights were not found: {deform_path}"
            )
        print(f"[load] deformation weights: {deform_path}")
        state = torch.load(str(deform_path), map_location="cuda")
        gs_model._deformation.deformation_net.load_state_dict(state, strict=True)
    else:
        if deform_path.exists():
            print(f"[info] {deform_path} exists, but --deform_on 0, so it is not used.")


def main() -> None:
    parser = ArgumentParser(description="Standalone mhravatar evaluator")
    parser.add_argument("--dat_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Training output directory")
    parser.add_argument("--configs", type=str, default=None, help='Optional config list, e.g. "a.yaml;b.yaml"')
    parser.add_argument("--pc_dir", type=str, default=None, help="Optional point_cloud/iteration_<N> directory")
    parser.add_argument("--iteration", type=int, default=-1, help="-1 means latest checkpoint")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--canonical_split", type=str, default="train")
    parser.add_argument("--canonical_index", type=int, default=0)
    parser.add_argument("--deform_on", type=int, default=0, choices=[0, 1])
    parser.add_argument("--white_background", type=int, default=None, choices=[0, 1])
    parser.add_argument("--total_iteration", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--allow_missing_embedding", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args, extras = parser.parse_known_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This project assumes CUDA; torch.cuda.is_available() is False.")

    model_path = _resolve_path(args.model_path, args.dat_dir)
    model_path = os.path.abspath(model_path)

    config_paths = _split_configs(args.configs)
    if not config_paths:
        saved_config = Path(model_path) / "config.yaml"
        if not saved_config.exists():
            raise FileNotFoundError(
                f"--configs was omitted, but saved config was not found: {saved_config}"
            )
        config_paths = [str(saved_config)]

    print(f"[config] {config_paths}")
    config = libcore.load_from_config(config_paths, cli_args=extras)
    config.dataset.dat_dir = args.dat_dir
    config.cache_dir = os.path.join(args.dat_dir, f"cache_{Path(config_paths[0]).stem}")

    if args.seed is not None:
        libcore.set_seed(args.seed)
    else:
        libcore.set_seed(config.get("seed", 9061))

    iteration, pc_dir = _resolve_checkpoint(model_path, args.pc_dir, args.iteration)
    total_iteration = args.total_iteration
    if total_iteration is None:
        total_iteration = int(config.get("optim", {}).get("total_iteration", iteration))

    if args.white_background is None:
        white_background = bool(config.get("optim", {}).get("white_background", False))
    else:
        white_background = bool(args.white_background)

    print(f"[data] canonical split={args.canonical_split}, eval split={args.split}")
    frameset_cano = make_frameset_data(config.dataset, split=args.canonical_split)
    frameset_eval = make_frameset_data(config.dataset, split=args.split)
    cano_batch = frameset_cano.__getitem__(args.canonical_index)
    cano_mesh = cano_batch["mesh_info"]

    print("[model] building SplattingAvatarModel")
    gs_model = SplattingAvatarModel(config.model, cano_mesh, args, verbose=True)
    gs_model.create_from_canonical(cano_mesh)
    _load_checkpoint(
        gs_model,
        pc_dir=pc_dir,
        deform_on=bool(args.deform_on),
        allow_missing_embedding=args.allow_missing_embedding,
    )
    gs_model.eval()

    pipe = config.pipe
    if args.out_dir is None:
        out_dir = Path(model_path) / f"eval_{iteration}_{args.split}"
    else:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = Path(model_path) / out_dir
    render_dir = out_dir / "render"
    compare_dir = out_dir / "compare"
    stats_fn = out_dir / "stats.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] iteration={iteration}, total_iteration={total_iteration}, deform_on={bool(args.deform_on)}")
    print(f"[eval] outputs: {out_dir}")

    with torch.no_grad():
        stats = testing_routine(
            current_time=0.0,
            tb_writer=None,
            pipe=pipe,
            iteration=iteration,
            total_iteration=total_iteration,
            frameset=frameset_eval,
            gs_model=gs_model,
            deform_on=bool(args.deform_on),
            white_background=white_background,
            render_dir=str(render_dir),
            compare_dir=str(compare_dir),
            verify=None,
        )

    with open(stats_fn, "w") as f:
        json.dump(stats, f, indent=2)

    print("[done]", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
