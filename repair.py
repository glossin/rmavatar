import numpy as np
from pathlib import Path

dat_dir = Path("/root/data1/datasets/PeopleSnapshot_MHR_RMAvatar/female-3-casual")

cam_path = dat_dir / "cameras.npz"
mhr_path = dat_dir / "mhr" / "model_params.npz"

old_cam = np.load(cam_path)
mp = np.load(mhr_path)

H = int(old_cam["height"])
W = int(old_cam["width"])

# model_params.npz 里 focal_length 是每帧一个，一般都相同
focal = float(np.asarray(mp["focal_length"]).reshape(-1)[0])

K_sam = np.array(
    [
        [focal, 0.0, W / 2.0],
        [0.0, focal, H / 2.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

E = np.eye(4, dtype=np.float32)

backup = dat_dir / "cameras_before_sam_fix.npz"
if not backup.exists():
    np.savez(
        backup,
        intrinsic=old_cam["intrinsic"],
        extrinsic=old_cam["extrinsic"],
        height=old_cam["height"],
        width=old_cam["width"],
    )

np.savez(
    cam_path,
    intrinsic=K_sam,
    extrinsic=E,
    height=old_cam["height"],
    width=old_cam["width"],
)

print("Saved fixed cameras.npz")
print("K_sam =\n", K_sam)
print("extrinsic =\n", E)
print("backup =", backup)