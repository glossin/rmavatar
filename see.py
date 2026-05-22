import os
import json
import numpy as np
import cv2

subj = "/root/data1/project/RMAvatar/data/PeopleSnapshot/female-3-casual"

with open(os.path.join(subj, "config.json"), "r") as f:
    cfg = json.load(f)

print("gender:", cfg["gender"])
print("train:", cfg["train"])
print("test:", cfg["test"])

cam = np.load(os.path.join(subj, "cameras.npz"))
print("K:", cam["intrinsic"].shape)
print("extrinsic:", cam["extrinsic"].shape)
print("H/W:", cam["height"], cam["width"])

poses = np.load(os.path.join(subj, "poses.npz"))
print("betas:", poses["betas"].shape)
print("thetas:", poses["thetas"].shape)
print("transl:", poses["transl"].shape)

img = cv2.imread(os.path.join(subj, "images/image_0000.png"))
mask = cv2.imread(os.path.join(subj, "masks/mask_0000.png"), cv2.IMREAD_UNCHANGED)
print("image:", img.shape)
print("mask:", mask.shape)