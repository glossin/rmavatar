import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from model import libcore
from dataset.mhr_avatar_reader import MHRInstantAvatarDataset


def read_camera_npz(cameras_npz: Path, invert_extrinsic: bool = False):
    data = np.load(str(cameras_npz))

    K = data["intrinsic"].astype(np.float32)
    extrinsic = data["extrinsic"].astype(np.float32)

    if extrinsic.shape == (3, 4):
        bottom = np.array([[0, 0, 0, 1]], dtype=np.float32)
        extrinsic = np.concatenate([extrinsic, bottom], axis=0)

    if extrinsic.shape != (4, 4):
        raise ValueError(f"Expected extrinsic shape [4,4] or [3,4], got {extrinsic.shape}")

    if invert_extrinsic:
        extrinsic = np.linalg.inv(extrinsic).astype(np.float32)

    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]

    if "height" in data and "width" in data:
        H = int(data["height"])
        W = int(data["width"])
    else:
        H = W = None

    return K, R, T, H, W


def project_vertices(verts, K, R, T):
    """
    verts: [N, 3] world/RMAvatar-space vertices
    K:     [3, 3]
    R,T:   world-to-camera
    """
    verts_cam = verts @ R.T + T[None, :]

    z = verts_cam[:, 2]
    proj = verts_cam @ K.T

    uv = np.zeros((verts.shape[0], 2), dtype=np.float32)
    valid = z > 1e-6

    uv[valid, 0] = proj[valid, 0] / proj[valid, 2]
    uv[valid, 1] = proj[valid, 1] / proj[valid, 2]

    return uv, z, valid


def draw_gt_mask_contour(img_bgr, mask):
    if mask is None:
        return img_bgr

    overlay = img_bgr.copy()
    mask_bin = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # GT mask contour: green
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)

    return overlay


def triangle_is_reasonable(pts, W, H, margin=2000):
    """
    避免投影飞掉的点导致 OpenCV 画超大三角形。
    """
    if np.any(~np.isfinite(pts)):
        return False

    x = pts[:, 0]
    y = pts[:, 1]

    if np.all(x < -margin) or np.all(x > W + margin):
        return False
    if np.all(y < -margin) or np.all(y > H + margin):
        return False

    return True


def rasterize_mesh_silhouette(uv, z, faces, H, W):
    mesh_mask = np.zeros((H, W), dtype=np.uint8)

    for f in faces:
        if not np.all(z[f] > 1e-6):
            continue

        pts = np.round(uv[f]).astype(np.int32)

        if not triangle_is_reasonable(pts, W, H):
            continue

        cv2.fillConvexPoly(mesh_mask, pts, 255)

    return mesh_mask


def draw_mesh_wireframe(img_bgr, uv, z, faces, face_stride=1, alpha=0.8):
    overlay = img_bgr.copy()
    H, W = img_bgr.shape[:2]

    # MHR wireframe: red
    color = (0, 0, 255)

    for f in faces[::face_stride]:
        if not np.all(z[f] > 1e-6):
            continue

        pts = np.round(uv[f]).astype(np.int32)

        if not triangle_is_reasonable(pts, W, H):
            continue

        cv2.polylines(
            overlay,
            [pts],
            isClosed=True,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    out = cv2.addWeighted(overlay, alpha, img_bgr, 1.0 - alpha, 0)
    return out


def blend_mesh_silhouette(img_bgr, mesh_mask, alpha=0.35):
    overlay = img_bgr.copy()

    # MHR silhouette fill: blue
    blue = np.zeros_like(img_bgr)
    blue[..., 0] = 255

    mask = mesh_mask > 0
    overlay[mask] = cv2.addWeighted(
        img_bgr[mask],
        1.0 - alpha,
        blue[mask],
        alpha,
        0,
    )

    return overlay


def compute_iou(mask_a, mask_b):
    a = mask_a > 0
    b = mask_b > 0

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 0.0

    return float(inter) / float(union)


def make_mask_compare(gt_mask, mesh_mask):
    """
    输出一张颜色图：
      green: GT mask only
      blue:  MHR mesh only
      white: overlap
    """
    H, W = gt_mask.shape[:2]
    out = np.zeros((H, W, 3), dtype=np.uint8)

    gt = gt_mask > 0
    mh = mesh_mask > 0

    overlap = gt & mh
    gt_only = gt & (~mh)
    mh_only = mh & (~gt)

    out[gt_only] = (0, 255, 0)      # green
    out[mh_only] = (255, 0, 0)      # blue
    out[overlap] = (255, 255, 255)  # white

    return out


def load_image_and_mask(dat_dir: Path, frm_idx: int):
    img_path = dat_dir / "images" / f"image_{frm_idx:04d}.png"
    mask_path = dat_dir / "masks" / f"mask_{frm_idx:04d}.png"

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    if mask.ndim == 3:
        mask = mask[..., 0]

    mask = (mask > 0).astype(np.uint8) * 255

    return img, mask, img_path, mask_path


def dump_one_frame(
    ds,
    dat_dir: Path,
    out_dir: Path,
    frm_idx: int,
    K,
    R,
    T,
    face_stride=1,
):
    img_bgr, gt_mask, img_path, _ = load_image_and_mask(dat_dir, frm_idx)

    H, W = img_bgr.shape[:2]

    mesh_info = ds.get_mhr_mesh(frm_idx)

    verts = mesh_info["mesh_verts"].detach().cpu().numpy().astype(np.float32)
    faces = mesh_info["mesh_faces"].detach().cpu().numpy().astype(np.int32)

    uv, z, valid = project_vertices(verts, K, R, T)

    mesh_mask = rasterize_mesh_silhouette(uv, z, faces, H, W)
    iou = compute_iou(gt_mask, mesh_mask)

    base = img_bgr.copy()
    base = draw_gt_mask_contour(base, gt_mask)

    wire = draw_mesh_wireframe(base, uv, z, faces, face_stride=face_stride)
    sil = blend_mesh_silhouette(img_bgr, mesh_mask)
    sil = draw_gt_mask_contour(sil, gt_mask)

    mask_compare = make_mask_compare(gt_mask, mesh_mask)

    cv2.putText(
        wire,
        f"frame={frm_idx}  mesh-mask IoU={iou:.4f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        sil,
        f"blue=MHR silhouette, green=GT mask contour, IoU={iou:.4f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    out_wire = out_dir / f"frame_{frm_idx:04d}_mhr_wire.jpg"
    out_sil = out_dir / f"frame_{frm_idx:04d}_mhr_silhouette.jpg"
    out_cmp = out_dir / f"frame_{frm_idx:04d}_mhr_mask_compare.jpg"

    cv2.imwrite(str(out_wire), wire)
    cv2.imwrite(str(out_sil), sil)
    cv2.imwrite(str(out_cmp), mask_compare)

    print(f"[OK] frame {frm_idx:04d}")
    print(f"     image: {img_path}")
    print(f"     verts: {verts.shape}, faces: {faces.shape}")
    print(f"     z range: {z.min():.6f} ~ {z.max():.6f}, valid_z={valid.mean():.4f}")
    print(f"     mesh-mask IoU: {iou:.6f}")
    print(f"     saved: {out_wire}")
    print(f"            {out_sil}")
    print(f"            {out_cmp}")


def parse_frame_ids(s):
    if s is None or s == "":
        return None

    out = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue

        if "-" in item:
            a, b = item.split("-")
            a = int(a)
            b = int(b)
            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(item))

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dat_dir", type=str, required=True)
    parser.add_argument(
        "--configs",
        type=lambda s: [i for i in s.split(";")],
        required=True,
        help="same as train_rmavatar.py, e.g. configs/mhr_avatar.yaml",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument(
        "--frame_ids",
        type=str,
        default=None,
        help="actual frame ids, e.g. 0,100,200 or 100-110. "
             "注意这是 image_XXXX.png 的 XXXX，不是 dataset idx。",
    )
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--face_stride", type=int, default=1)

    # 用来快速排查坐标系问题
    parser.add_argument(
        "--invert_extrinsic",
        action="store_true",
        help="If cameras.npz extrinsic is actually c2w, try this.",
    )
    parser.add_argument(
        "--disable_pred_cam_t",
        action="store_true",
        help="Temporarily ignore pred_cam_t in mhr/model_params.npz.",
    )
    parser.add_argument(
        "--force_coord_fix",
        type=int,
        default=-1,
        help="-1: use config; 0: disable _mhr_to_rmavatar_space coord fix; 1: enable it.",
    )

    args, extras = parser.parse_known_args()

    dat_dir = Path(args.dat_dir).resolve()

    if args.out_dir is None:
        out_dir = dat_dir / "debug_mhr_overlay"
    else:
        out_dir = Path(args.out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    config = libcore.load_from_config(args.configs, cli_args=extras)
    config.dataset.dat_dir = str(dat_dir)

    ds = MHRInstantAvatarDataset(config.dataset, split=args.split)

    if args.disable_pred_cam_t:
        print("[DEBUG] disable pred_cam_t")
        ds.pred_cam_t = None
        ds.mhr_verts = None

    if args.force_coord_fix in [0, 1]:
        ds.use_mhr_coord_fix = bool(args.force_coord_fix)
        ds.mhr_verts = None
        print(f"[DEBUG] force use_mhr_coord_fix = {ds.use_mhr_coord_fix}")

    K, R, T, H, W = read_camera_npz(
        dat_dir / "cameras.npz",
        invert_extrinsic=args.invert_extrinsic,
    )

    frame_ids = parse_frame_ids(args.frame_ids)

    if frame_ids is None:
        # 默认取 split 里的前、中、后几帧
        frm_list = list(ds.frm_list)
        if len(frm_list) <= 5:
            frame_ids = frm_list
        else:
            idxs = np.linspace(0, len(frm_list) - 1, 5).astype(int).tolist()
            frame_ids = [frm_list[i] for i in idxs]

    print("[INFO] output dir:", out_dir)
    print("[INFO] frame_ids:", frame_ids)
    print("[INFO] K:\n", K)
    print("[INFO] R:\n", R)
    print("[INFO] T:", T)

    for frm_idx in frame_ids:
        dump_one_frame(
            ds=ds,
            dat_dir=dat_dir,
            out_dir=out_dir,
            frm_idx=frm_idx,
            K=K,
            R=R,
            T=T,
            face_stride=max(1, args.face_stride),
        )


if __name__ == "__main__":
    main()