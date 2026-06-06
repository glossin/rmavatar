"""
python merge_mhr_refined_params.py \
  --dat_dir /root/data1/project/rmavatar-mhr/datasets/PeopleSnapshot/111/male-3-casual \
  --train_refined /root/data1/project/rmavatar-mhr/out/mhr_round1/model_params_refined_train_predcam.npz \
  --test_refined /root/data1/project/rmavatar-mhr/out/mhr_round1/model_params_refined_test_predcam.npz \
  --out data1/project/rmavatar-mhr/out/mhr_round1/model_params_refined.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np


FRAME_KEYS = [
    "model_parameters",
    "mhr_model_params",
    "expression_coeffs",
    "expr_params",
    "pred_cam_t",
]

GLOBAL_KEYS = [
    "identity_coeffs",
    "focal_length",
]


def get_frame_ids(dat_dir: Path, split: str):
    with open(dat_dir / "config.json", "r") as f:
        cfg = json.load(f)

    if split not in cfg:
        raise KeyError(f"split '{split}' not found in {dat_dir / 'config.json'}")

    s = cfg[split]
    return np.arange(
        int(s["start"]),
        int(s["end"]) + 1,
        int(s.get("skip", 1)),
        dtype=np.int64,
    )


def copy_split_values(out, src, split_ids, total_frames, split_name):
    """
    out:
        dict copied from base full-length params
    src:
        refined npz dict
    split_ids:
        global frame ids for this split
    total_frames:
        full sequence length

    Supports:
        src[key].shape[0] == total_frames: full-length refined file
        src[key].shape[0] == len(split_ids): split-local refined file
    """
    n_split = len(split_ids)

    for key in FRAME_KEYS:
        if key not in src.files:
            continue

        value = src[key]

        if value.ndim == 0:
            continue

        if value.shape[0] == total_frames:
            # full-length refined npz
            refined_values = value[split_ids]
        elif value.shape[0] == n_split:
            # split-local refined npz
            refined_values = value
        else:
            raise ValueError(
                f"{split_name}: key '{key}' has first dim {value.shape[0]}, "
                f"expected either total_frames={total_frames} or split_frames={n_split}."
            )

        # Some aliases may not exist in base; create them if source has them.
        if key not in out:
            out[key] = np.zeros(
                (total_frames,) + refined_values.shape[1:],
                dtype=refined_values.dtype,
            )

        out[key][split_ids] = refined_values.astype(out[key].dtype, copy=False)

    return out


def sync_aliases(out):
    """
    Keep aliases consistent:
      model_parameters <-> mhr_model_params
      expression_coeffs <-> expr_params
    """
    if "model_parameters" in out:
        out["mhr_model_params"] = out["model_parameters"].copy()
    elif "mhr_model_params" in out:
        out["model_parameters"] = out["mhr_model_params"].copy()

    if "expression_coeffs" in out:
        out["expr_params"] = out["expression_coeffs"].copy()
    elif "expr_params" in out:
        out["expression_coeffs"] = out["expr_params"].copy()

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dat_dir", required=True)
    parser.add_argument("--base", default=None, help="default: <dat_dir>/mhr/model_params.npz")
    parser.add_argument("--train_refined", default=None)
    parser.add_argument("--test_refined", default=None)
    parser.add_argument("--val_refined", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dat_dir = Path(args.dat_dir)

    base_path = Path(args.base) if args.base else dat_dir / "mhr" / "model_params.npz"
    out_path = Path(args.out)

    if not out_path.is_absolute():
        out_path = dat_dir / out_path

    base = np.load(base_path)

    if "model_parameters" in base.files:
        total_frames = base["model_parameters"].shape[0]
    elif "mhr_model_params" in base.files:
        total_frames = base["mhr_model_params"].shape[0]
    else:
        raise KeyError("base npz must contain model_parameters or mhr_model_params")

    # Copy all base keys first, including metadata.
    out = {}
    for key in base.files:
        out[key] = base[key].copy()

    split_to_file = {
        "train": args.train_refined,
        "test": args.test_refined,
        "val": args.val_refined,
    }

    for split, refined_path in split_to_file.items():
        if refined_path is None:
            continue

        refined_path = Path(refined_path)
        if not refined_path.is_absolute():
            refined_path = dat_dir / refined_path

        if not refined_path.exists():
            raise FileNotFoundError(refined_path)

        split_ids = get_frame_ids(dat_dir, split)
        refined = np.load(refined_path)

        print(f"[merge] split={split}")
        print(f"        ids: {split_ids[:5]} ... {split_ids[-5:]}, n={len(split_ids)}")
        print(f"        refined: {refined_path}")

        out = copy_split_values(
            out=out,
            src=refined,
            split_ids=split_ids,
            total_frames=total_frames,
            split_name=split,
        )

    out = sync_aliases(out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)

    print(f"[write] {out_path}")

    check = np.load(out_path)
    for k in ["model_parameters", "mhr_model_params", "expression_coeffs", "expr_params", "pred_cam_t", "identity_coeffs"]:
        if k in check.files:
            print(f"  {k}: {check[k].shape} {check[k].dtype}")


if __name__ == "__main__":
    main()