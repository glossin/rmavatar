import os
import csv
import json
from pathlib import Path
from argparse import ArgumentParser, Namespace

import cv2
import numpy as np
import torch
from tqdm import tqdm

from model import libcore
from model.rmavatar_model import SplattingAvatarModel
from dataset.dataset_helper import make_frameset_data, make_dataloader
from utils.metrics import img_mse, img_ssim, img_psnr, perceptual


def resolve_path(dat_dir, path):
    """
    Resolve path robustly.

    Priority:
    1. absolute path
    2. existing relative path from current working directory
    3. relative to dat_dir
    """
    if path is None:
        return None

    path = str(path)
    p = Path(path)

    if p.is_absolute():
        return str(p)

    if p.exists():
        return str(p.resolve())

    return str(Path(dat_dir) / p)


def to_int(value):
    if torch.is_tensor(value):
        return int(value.detach().cpu().view(-1)[0].item())
    return int(value)


def move_mesh_info_to_cuda(mesh_info):
    out = {}

    for k, v in mesh_info.items():
        if torch.is_tensor(v):
            out[k] = v.cuda(non_blocking=True)
        else:
            out[k] = v

    return out


def tensor_to_uint8_rgb(image):
    """
    image: torch tensor, shape (3, H, W), range [0, 1]
    return: numpy uint8 RGB, shape (H, W, 3)
    """
    image = image.detach().float().cpu().clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255.0).round().astype(np.uint8)
    return image


def make_compare_image(gt_image, pred_image, psnr_value, ssim_value, lpips_value):
    """
    Create side-by-side compare image:
    GT | Render | Error heatmap
    Output is BGR for cv2.imwrite.
    """
    gt_rgb = tensor_to_uint8_rgb(gt_image)
    pred_rgb = tensor_to_uint8_rgb(pred_image)

    error = torch.abs(gt_image.detach() - pred_image.detach()).mean(dim=0)
    error = error.float().cpu().numpy()

    if error.max() > 1.0e-8:
        error = error / error.max()

    error_u8 = (error * 255.0).round().astype(np.uint8)
    error_color = cv2.applyColorMap(error_u8, cv2.COLORMAP_JET)
    error_rgb = cv2.cvtColor(error_color, cv2.COLOR_BGR2RGB)

    canvas_rgb = np.concatenate([gt_rgb, pred_rgb, error_rgb], axis=1)

    header_h = 42
    h, w, _ = canvas_rgb.shape
    header = np.ones((header_h, w, 3), dtype=np.uint8) * 255

    text = f"GT | Render | Error    PSNR={psnr_value:.3f}  SSIM={ssim_value:.4f}"

    if not np.isnan(lpips_value):
        text += f"  LPIPS={lpips_value:.4f}"

    cv2.putText(
        header,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    out_rgb = np.concatenate([header, canvas_rgb], axis=0)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return out_bgr


def write_tensor_image(path, image, rgb2bgr=True):
    """
    Save torch tensor image.
    image: (3, H, W), [0, 1]
    """
    image_rgb = tensor_to_uint8_rgb(image)

    if rgb2bgr:
        image_out = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    else:
        image_out = image_rgb

    cv2.imwrite(path, image_out)


def compute_fg_psnr(image, gt_image, gt_alpha_mask):
    """
    Foreground PSNR computed only inside gt_alpha_mask.
    image / gt_image: (3, H, W)
    gt_alpha_mask: (1, H, W) or (H, W)
    """
    if gt_alpha_mask is None:
        return float("nan")

    mask = gt_alpha_mask

    if not torch.is_tensor(mask):
        return float("nan")

    mask = mask.to(image.device).float()

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    if mask.dim() == 3 and mask.shape[0] != 1:
        mask = mask[:1]

    mask = (mask > 0.5).float()

    denom = mask.sum() * image.shape[0]

    if denom.item() <= 1:
        return float("nan")

    mse = ((image - gt_image).pow(2) * mask).sum() / (denom + 1.0e-8)
    psnr = -10.0 * torch.log10(mse + 1.0e-8)

    return float(psnr.detach().cpu().item())


def mean_or_nan(values):
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return float("nan")

    return float(np.nanmean(values))


def check_dataset_dir(dat_dir):
    required = [
        os.path.join(dat_dir, "config.json"),
        os.path.join(dat_dir, "poses.npz"),
        os.path.join(dat_dir, "images"),
        os.path.join(dat_dir, "masks"),
    ]

    missing = [p for p in required if not os.path.exists(p)]

    if len(missing) > 0:
        raise FileNotFoundError(
            "[Eval] --dat_dir must point to the dataset subject directory, "
            "not the model output directory.\n"
            f"Current --dat_dir: {dat_dir}\n"
            "Missing:\n  " + "\n  ".join(missing)
        )


def check_checkpoint_dir(pc_dir, deform_on):
    required = [
        os.path.join(pc_dir, "point_cloud.ply"),
        os.path.join(pc_dir, "embedding.json"),
    ]

    if deform_on:
        required.append(os.path.join(pc_dir, "deform.pth"))

    missing = [p for p in required if not os.path.exists(p)]

    if len(missing) > 0:
        raise FileNotFoundError(
            "[Eval] Invalid checkpoint directory.\n"
            f"pc_dir: {pc_dir}\n"
            "Missing:\n  " + "\n  ".join(missing)
        )


if __name__ == "__main__":
    parser = ArgumentParser("Standalone RMAvatar evaluation")

    parser.add_argument("--dat_dir", type=str, required=True)
    parser.add_argument(
        "--configs",
        type=lambda s: [i for i in s.split(";")],
        required=True,
        help=(
            "Base config files, for example: "
            "configs/peoplesnapshot.yaml;configs/instant_avatar.yaml"
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "val"],
        help="Dataset split to evaluate.",
    )

    parser.add_argument(
        "--iteration",
        type=int,
        required=True,
        help="Checkpoint iteration, for example 50000.",
    )

    parser.add_argument(
        "--total_iteration",
        type=int,
        default=None,
        help="Usually same as --iteration. If omitted, use --iteration.",
    )

    parser.add_argument(
        "--pc_dir",
        type=str,
        default=None,
        help=(
            "Path to checkpoint iteration directory, e.g. "
            ".../point_cloud/iteration_50000. "
            "If provided, this has priority over --model_path."
        ),
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="output-splatting/baseline+offset(5w)",
        help=(
            "Training output root directory. Used only when --pc_dir is not provided. "
            "The script will append point_cloud/iteration_xxxxx."
        ),
    )

    parser.add_argument(
        "--deform_on",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use 1 if the checkpoint was trained with deform_on=1.",
    )

    parser.add_argument(
        "--white_background",
        type=int,
        default=1,
        choices=[0, 1],
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Evaluation output directory. If omitted, auto-create under model_path.",
    )

    parser.add_argument(
        "--save_images",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to save render / gt / compare images.",
    )

    parser.add_argument(
        "--compute_lpips",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to compute LPIPS. Set 0 for faster evaluation.",
    )

    args, extras = parser.parse_known_args()

    check_dataset_dir(args.dat_dir)

    config = libcore.load_from_config(args.configs, cli_args=extras)
    config.dataset.dat_dir = args.dat_dir
    config.cache_dir = os.path.join(
        args.dat_dir,
        f"cache_{Path(args.configs[0]).stem}",
    )

    split = args.split
    iteration = int(args.iteration)
    total_iteration = int(args.total_iteration) if args.total_iteration is not None else iteration

    deform_on = bool(args.deform_on)
    white_background = bool(args.white_background)
    save_images = bool(args.save_images)
    compute_lpips = bool(args.compute_lpips)

    if args.pc_dir is not None:
        pc_dir = resolve_path(args.dat_dir, args.pc_dir)
        model_path = str(Path(pc_dir).parents[1])
    else:
        model_path = resolve_path(args.dat_dir, args.model_path)
        pc_dir = os.path.join(
            model_path,
            "point_cloud",
            f"iteration_{iteration}",
        )

    check_checkpoint_dir(pc_dir, deform_on=deform_on)

    if args.out_dir is None:
        out_dir = os.path.join(
            model_path,
            f"standalone_eval_{split}_{iteration}",
        )
    else:
        out_dir = resolve_path(args.dat_dir, args.out_dir)

    render_dir = os.path.join(out_dir, "render")
    gt_dir = os.path.join(out_dir, "gt")
    compare_dir = os.path.join(out_dir, "compare")

    os.makedirs(out_dir, exist_ok=True)

    if save_images:
        os.makedirs(render_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(compare_dir, exist_ok=True)

    print(f"[Eval] dat_dir: {args.dat_dir}")
    print(f"[Eval] configs: {args.configs}")
    print(f"[Eval] split: {split}")
    print(f"[Eval] iteration: {iteration}")
    print(f"[Eval] total_iteration: {total_iteration}")
    print(f"[Eval] pc_dir: {pc_dir}")
    print(f"[Eval] model_path: {model_path}")
    print(f"[Eval] out_dir: {out_dir}")
    print(f"[Eval] deform_on: {deform_on}")
    print(f"[Eval] white_background: {white_background}")
    print(f"[Eval] save_images: {save_images}")
    print(f"[Eval] compute_lpips: {compute_lpips}")

    # Dataset
    frameset_train = make_frameset_data(config.dataset, split="train")
    frameset_eval = make_frameset_data(config.dataset, split=split)
    dataloader = make_dataloader(frameset_eval, shuffle=False)

    # Canonical mesh must match training.
    first_batch = frameset_train.__getitem__(0)
    cano_mesh = first_batch["mesh_info"]

    # Model
    pipe = config.pipe

    model_args = Namespace(
        dat_dir=args.dat_dir,
        deform_on=int(deform_on),
        model_path=model_path,
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

    gs_model.load_ply(ply_fn)
    gs_model.load_from_embedding(embed_fn)

    if os.path.exists(deform_fn):
        gs_model.load_deform_weights(pc_dir, iteration=iteration)
    elif deform_on:
        raise FileNotFoundError(
            f"deform_on=True, but deform.pth does not exist: {deform_fn}"
        )

    gs_model.eval()

    rows = []

    psnr_values = []
    ssim_values = []
    lpips_values = []
    fg_psnr_values = []

    current_time = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Evaluating [{split}]")

        for batches in pbar:
            batch = batches[0]

            local_idx = to_int(batch["idx"])

            if "frm_idx" in batch:
                frm_idx = to_int(batch["frm_idx"])
            else:
                frm_idx = local_idx

            mesh_info = move_mesh_info_to_cuda(batch["mesh_info"])
            pose = mesh_info.get("pose", None)

            gs_model.update_to_cano_mesh(mesh_info)

            viewpoint_cam = batch["scene_cameras"][0].cuda()
            viewpoint_cam.time = current_time

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

            image = render_pkg["render"].clamp(0, 1)
            gt_image = render_pkg["gt_image"].clamp(0, 1)
            gt_alpha_mask = render_pkg.get("gt_alpha_mask", None)

            rmse = img_mse(
                image[None, ...],
                gt_image[None, ...],
                mask=None,
                error_type="rmse",
                use_mask=False,
            )

            psnr = img_psnr(
                image[None, ...],
                gt_image[None, ...],
                rmse=rmse,
            )

            ssim = img_ssim(
                image[None, ...],
                gt_image[None, ...],
            )

            if compute_lpips:
                lpips_value = perceptual(
                    image[None, ...],
                    gt_image[None, ...],
                    mask=None,
                    use_mask=False,
                ).mean()
                lpips_float = float(lpips_value.detach().cpu().item())
            else:
                lpips_float = float("nan")

            psnr_float = float(psnr.detach().cpu().item())
            ssim_float = float(ssim.detach().cpu().item())
            fg_psnr_float = compute_fg_psnr(
                image,
                gt_image,
                gt_alpha_mask,
            )

            psnr_values.append(psnr_float)
            ssim_values.append(ssim_float)
            lpips_values.append(lpips_float)
            fg_psnr_values.append(fg_psnr_float)

            row = {
                "local_idx": local_idx,
                "frm_idx": frm_idx,
                "psnr": psnr_float,
                "ssim": ssim_float,
                "lpips": lpips_float,
                "fg_psnr": fg_psnr_float,
            }
            rows.append(row)

            pbar.set_postfix({
                "psnr": f"{mean_or_nan(psnr_values):.2f}",
                "ssim": f"{mean_or_nan(ssim_values):.4f}",
                "lpips": f"{mean_or_nan(lpips_values):.4f}",
                "fg_psnr": f"{mean_or_nan(fg_psnr_values):.2f}",
            })

            if save_images:
                filename = f"{local_idx:04d}_frame_{frm_idx:05d}"

                write_tensor_image(
                    os.path.join(render_dir, f"{filename}.png"),
                    image,
                    rgb2bgr=True,
                )

                write_tensor_image(
                    os.path.join(gt_dir, f"{filename}.png"),
                    gt_image,
                    rgb2bgr=True,
                )

                compare = make_compare_image(
                    gt_image,
                    image,
                    psnr_float,
                    ssim_float,
                    lpips_float,
                )

                cv2.imwrite(
                    os.path.join(compare_dir, f"{filename}.jpg"),
                    compare,
                )

    metrics_csv = os.path.join(out_dir, "metrics.csv")

    with open(metrics_csv, "w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "local_idx",
                "frm_idx",
                "psnr",
                "ssim",
                "lpips",
                "fg_psnr",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "split": split,
        "iteration": iteration,
        "total_iteration": total_iteration,
        "num_frames": len(rows),
        "psnr": mean_or_nan(psnr_values),
        "ssim": mean_or_nan(ssim_values),
        "lpips": mean_or_nan(lpips_values),
        "fg_psnr": mean_or_nan(fg_psnr_values),
        "pc_dir": pc_dir,
        "model_path": model_path,
        "out_dir": out_dir,
        "deform_on": deform_on,
        "white_background": white_background,
    }

    summary_json = os.path.join(out_dir, "summary.json")

    with open(summary_json, "w") as fp:
        json.dump(summary, fp, indent=2)

    print("[Eval] done")
    print(f"[Eval] metrics_csv: {metrics_csv}")
    print(f"[Eval] summary_json: {summary_json}")
    print(json.dumps(summary, indent=2))