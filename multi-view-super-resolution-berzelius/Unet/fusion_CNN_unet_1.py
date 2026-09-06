#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("start fusion_CNN")

import os
import ast
import gc
import json
import time
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from generative.losses import PerceptualLoss

from WarvitoCodes.models_unet_v2_conditioned_AdaDM_2D import UNetModel


# =========================================================
# ---------------------- argparse -------------------------
# =========================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Three-axis UNet inference + 3D CNN fusion training"
    )

    # ---------- axis 0 ----------
    p.add_argument("--axis0_pt", type=str, required=True)
    p.add_argument("--axis0_param", type=str, required=True)
    p.add_argument("--axis0_inputs", type=str, required=True)
    p.add_argument("--axis0_outputs", type=str, required=True)
    p.add_argument("--axis0_contexts", type=str, default=None)
    p.add_argument("--axis0_ranges", type=str, default=None)

    # ---------- axis 1 ----------
    p.add_argument("--axis1_pt", type=str, required=True)
    p.add_argument("--axis1_param", type=str, required=True)
    p.add_argument("--axis1_inputs", type=str, required=True)
    p.add_argument("--axis1_outputs", type=str, required=True)
    p.add_argument("--axis1_contexts", type=str, default=None)
    p.add_argument("--axis1_ranges", type=str, default=None)

    # ---------- axis 2 ----------
    p.add_argument("--axis2_pt", type=str, required=True)
    p.add_argument("--axis2_param", type=str, required=True)
    p.add_argument("--axis2_inputs", type=str, required=True)
    p.add_argument("--axis2_outputs", type=str, required=True)
    p.add_argument("--axis2_contexts", type=str, default=None)
    p.add_argument("--axis2_ranges", type=str, default=None)

    # output
    p.add_argument("--out_dir", type=str, required=True)

    # canonical gt axis
    p.add_argument("--canonical_axis", type=int, default=0, choices=[0, 1, 2])

    # optional reference nifti for affine/header
    p.add_argument("--ref_nifti_dir", type=str, default=None)
    p.add_argument("--ref_nifti_pattern", type=str, default="{subject}.nii.gz")
    p.add_argument("--save_nifti", action="store_true")

    # inference
    p.add_argument("--infer_batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)

    # fusion training
    p.add_argument("--fusion_epochs", type=int, default=40)
    p.add_argument("--fusion_lr", type=float, default=1e-4)
    p.add_argument("--fusion_batch_size", type=int, default=2)
    p.add_argument("--patch_size", type=int, nargs=3, default=[64, 64, 64])
    p.add_argument("--patches_per_subject", type=int, default=16)

    # fusion loss
    p.add_argument("--fusion_use_perceptual_loss", action="store_true")
    p.add_argument("--fusion_perceptual_weight", type=float, default=0.001)

    # misc
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=114514)
    p.add_argument("--max_subjects", type=int, default=None)

    return p.parse_args()


# =========================================================
# ---------------------- helpers --------------------------
# =========================================================

def mkdir(path):
    os.makedirs(path, exist_ok=True)


def set_seed(seed=114514):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_settings(param_path):
    with open(param_path, "r") as f:
        first_line = f.readline().strip()
    return ast.literal_eval(first_line)


def safe_torch_load(path, map_location="cpu"):
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    return ckpt


def infer_subject_count(inputs_path, n_slices):
    x = np.load(inputs_path, mmap_mode="r")
    n_total = x.shape[0]
    if n_total % n_slices != 0:
        raise ValueError(
            f"{inputs_path}: total slices = {n_total}, not divisible by n_slices = {n_slices}"
        )
    return n_total // n_slices


def get_axis_cfgs(args):
    return {
        0: {
            "pt": args.axis0_pt,
            "param": args.axis0_param,
            "inputs": args.axis0_inputs,
            "outputs": args.axis0_outputs,
            "contexts": args.axis0_contexts,
            "ranges": args.axis0_ranges,
        },
        1: {
            "pt": args.axis1_pt,
            "param": args.axis1_param,
            "inputs": args.axis1_inputs,
            "outputs": args.axis1_outputs,
            "contexts": args.axis1_contexts,
            "ranges": args.axis1_ranges,
        },
        2: {
            "pt": args.axis2_pt,
            "param": args.axis2_param,
            "inputs": args.axis2_inputs,
            "outputs": args.axis2_outputs,
            "contexts": args.axis2_contexts,
            "ranges": args.axis2_ranges,
        }
    }


def compute_padder(slice_size, channel_mult, use_initial_down):
    n_d1, n_d2 = slice_size
    n_down = len(channel_mult) - 1 + int(use_initial_down)

    if n_d1 % (2 ** n_down):
        r1 = 2 ** n_down - n_d1 % (2 ** n_down)
    else:
        r1 = 0

    if n_d2 % (2 ** n_down):
        r2 = 2 ** n_down - n_d2 % (2 ** n_down)
    else:
        r2 = 0

    padder = (r2 // 2, r2 // 2 + r2 % 2, r1 // 2, r1 // 2 + r1 % 2)
    return padder, r1, r2


def unpad_2d_batch(x, r1, r2):
    if r1 != 0 and r2 != 0:
        return x[:, :, r1 // 2:-(r1 // 2 + r1 % 2), r2 // 2:-(r2 // 2 + r2 % 2)]
    elif r1 != 0:
        return x[:, :, r1 // 2:-(r1 // 2 + r1 % 2), :]
    elif r2 != 0:
        return x[:, :, :, r2 // 2:-(r2 // 2 + r2 % 2)]
    else:
        return x


def stacked_slices_to_canonical(vol_stack, slice_dim):
    """
    vol_stack: (n_slices, H, W)
    canonical: (D0, D1, D2)
    """
    if slice_dim == 0:
        return vol_stack
    elif slice_dim == 1:
        return np.transpose(vol_stack, (1, 0, 2))
    elif slice_dim == 2:
        return np.transpose(vol_stack, (1, 2, 0))
    else:
        raise ValueError(f"Invalid slice_dim={slice_dim}")


def get_ref_nifti(subject_idx, ref_nifti_dir, ref_pattern):
    if ref_nifti_dir is None:
        return None
    path = os.path.join(ref_nifti_dir, ref_pattern.format(subject=subject_idx))
    if not os.path.exists(path):
        return None
    return nib.load(path)


def save_volume(volume, save_path_npy, save_nifti=False, ref_img=None):
    np.save(save_path_npy, volume.astype(np.float32))
    if save_nifti:
        if ref_img is not None:
            img = nib.Nifti1Image(volume.astype(np.float32), ref_img.affine, ref_img.header)
        else:
            img = nib.Nifti1Image(volume.astype(np.float32), np.eye(4))
        nib.save(img, str(Path(save_path_npy).with_suffix(".nii.gz")))


def build_unet_from_settings(settings):
    model = UNetModel(
        image_size=tuple(settings["slice_size"]),
        in_channels=settings["n_in_slices"],
        out_channels=1,
        model_channels=settings["n_channels"],
        num_res_blocks=settings["n_res_block"],
        attention_resolutions=settings["attention_res"],
        context_dim=settings.get("c_dim", None),
        dropout=settings["dropout"],
        use_ada=settings["use_ada"],
        channel_mult=tuple(settings["channel_mult"]),
        use_spatial_transformer=settings.get("use_spatial_transformer", False),
        use_mid_attention=settings.get("use_mid_attention", False),
        use_initial_downsample=settings.get("use_initial_down", False),
        num_groups=settings.get("num_groups", 16),
    )
    return model


def load_unet_model(pt_path, param_path, device):
    settings = load_settings(param_path)
    ckpt = safe_torch_load(pt_path, map_location="cpu")

    model = build_unet_from_settings(settings)

    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model = nn.DataParallel(model)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model, settings, ckpt


def normalize_conditions_list(conditions):
    if conditions is None:
        return []
    if isinstance(conditions, list):
        return conditions
    return list(conditions)


def infer_use_condition_from_settings(settings):
    conditions = normalize_conditions_list(settings.get("conditions", []))
    c_dim = settings.get("c_dim", None)
    use_st = settings.get("use_spatial_transformer", False)

    use_condition = (len(conditions) > 0) or (c_dim not in [None, 0, "None"])
    return use_condition, conditions, c_dim, use_st


def load_context_array(context_path):
    if context_path is None:
        return None
    if not os.path.exists(context_path):
        raise FileNotFoundError(f"contexts file not found: {context_path}")
    return np.load(context_path, mmap_mode="r")


def prepare_context_batch(contexts_subj, settings):
    """
    contexts_subj expected from original data: [S, 4, 1].
    Training used contexts[:, :2, :] and contexts[:, 3:4, :],
    giving final shape [S, 3, 1].
    """
    if contexts_subj is None:
        return None

    contexts_subj = np.asarray(contexts_subj, dtype=np.float32)

    if contexts_subj.ndim == 3 and contexts_subj.shape[1] == 4:
        contexts_subj = np.concatenate(
            (contexts_subj[:, :2, :], contexts_subj[:, 3:4, :]),
            axis=1
        )

    return torch.from_numpy(contexts_subj).float()


def unet_forward(model, x, timesteps, context=None, use_condition=False):
    """
    Compatible with both conditioned and unconditioned forward calls.
    """
    if use_condition and context is not None:
        try:
            return model(x, timesteps=timesteps, context=context)
        except TypeError:
            try:
                return model(x, timesteps, context)
            except TypeError:
                return model(x, timesteps=timesteps)
    else:
        return model(x, timesteps=timesteps)


# =========================================================
# ---------------- axis inference -------------------------
# =========================================================

def run_axis_inference_for_all_subjects(
    axis_id,
    axis_cfg,
    out_dir,
    device,
    infer_batch_size=16,
    max_subjects=None,
    ref_nifti_dir=None,
    ref_pattern="{subject}.nii.gz",
    save_nifti=False
):
    axis_out_dir = os.path.join(out_dir, f"axis{axis_id}")
    pred_dir = os.path.join(axis_out_dir, "pred")
    gt_dir = os.path.join(axis_out_dir, "gt_from_outputs")
    mkdir(axis_out_dir)
    mkdir(pred_dir)
    mkdir(gt_dir)

    model, settings, ckpt = load_unet_model(axis_cfg["pt"], axis_cfg["param"], device)

    use_condition, conditions, c_dim, use_st = infer_use_condition_from_settings(settings)

    n_slices = int(settings["n_slices"])
    slice_dim = int(settings["slice_dim"])
    slice_size = tuple(settings["slice_size"])

    padder, r1, r2 = compute_padder(
        slice_size=slice_size,
        channel_mult=tuple(settings["channel_mult"]),
        use_initial_down=settings.get("use_initial_down", False),
    )

    x = np.load(axis_cfg["inputs"], mmap_mode="r")

    y = np.load(axis_cfg["outputs"], mmap_mode="r")
    y = np.array(y, copy=True)
    y[y > 1.5] = 1.5

    contexts = load_context_array(axis_cfg["contexts"])

    n_subjects = x.shape[0] // n_slices
    if max_subjects is not None:
        n_subjects = min(n_subjects, max_subjects)

    idx_test = ckpt.get("idx_test", None)
    if idx_test is not None:
        idx_test = np.array(idx_test).astype(int)

    print(f"[axis {axis_id}] use_condition = {use_condition}")
    print(f"[axis {axis_id}] conditions = {conditions}")
    print(f"[axis {axis_id}] c_dim = {c_dim}")
    print(f"[axis {axis_id}] n_subjects = {n_subjects}, n_slices = {n_slices}, slice_dim = {slice_dim}")
    print(f"[axis {axis_id}] x shape = {x.shape}, y shape = {y.shape}")

    if use_condition and contexts is None:
        raise ValueError(
            f"Axis {axis_id} settings indicate condition is used, but contexts file is missing."
        )

    with torch.no_grad():
        for subj in range(n_subjects):
            s0 = subj * n_slices
            s1 = (subj + 1) * n_slices

            x_subj = torch.from_numpy(x[s0:s1]).float()
            y_subj = y[s0:s1]

            x_subj = F.pad(x_subj, padder, "constant", 0)

            if use_condition:
                c_subj = prepare_context_batch(contexts[s0:s1], settings)
            else:
                c_subj = None

            pred_slices = []

            for b0 in range(0, n_slices, infer_batch_size):
                b1 = min(b0 + infer_batch_size, n_slices)

                xb = x_subj[b0:b1].to(device)
                sb = xb.shape[0]
                timesteps = torch.ones(sb, dtype=torch.float32, device=xb.device)

                if c_subj is not None:
                    cb = c_subj[b0:b1].to(device)
                else:
                    cb = None

                pred = unet_forward(
                    model=model,
                    x=xb,
                    timesteps=timesteps,
                    context=cb,
                    use_condition=use_condition
                )

                pred = pred.detach().cpu()
                pred = unpad_2d_batch(pred, r1, r2)
                pred = pred[:, 0].numpy()

                pred_slices.append(pred)

                del xb, timesteps, cb, pred

            pred_stack = np.concatenate(pred_slices, axis=0)
            gt_stack = y_subj[:, 0] if (y_subj.ndim == 4 and y_subj.shape[1] == 1) else np.squeeze(y_subj)

            pred_vol = stacked_slices_to_canonical(pred_stack, slice_dim)
            gt_vol = stacked_slices_to_canonical(gt_stack, slice_dim)

            ref_img = get_ref_nifti(subj, ref_nifti_dir, ref_pattern)

            pred_path = os.path.join(pred_dir, f"subject_{subj:04d}.npy")
            gt_path = os.path.join(gt_dir, f"subject_{subj:04d}.npy")

            save_volume(pred_vol, pred_path, save_nifti=save_nifti, ref_img=ref_img)
            save_volume(gt_vol, gt_path, save_nifti=save_nifti, ref_img=ref_img)

            if subj % 5 == 0 or subj == n_subjects - 1:
                print(
                    f"[axis {axis_id}] subject {subj}/{n_subjects-1}, "
                    f"pred_vol shape = {pred_vol.shape}"
                )

            del x_subj, y_subj, c_subj, pred_stack, gt_stack, pred_vol, gt_vol
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    info = {
        "axis_id": axis_id,
        "settings": settings,
        "n_subjects": int(n_subjects),
        "n_slices": int(n_slices),
        "slice_dim": int(slice_dim),
        "slice_size": list(slice_size),
        "use_condition": bool(use_condition),
        "conditions": conditions,
        "c_dim": c_dim,
        "idx_test": None if idx_test is None else idx_test.tolist(),
        "pred_dir": pred_dir,
        "gt_dir": gt_dir,
    }

    with open(os.path.join(axis_out_dir, "axis_inference_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    return info


# =========================================================
# ---------------- fusion dataset -------------------------
# =========================================================

class FusionPatchDataset(Dataset):
    def __init__(
        self,
        subject_ids,
        pred_dirs,
        gt_dir,
        patch_size=(64, 64, 64),
        patches_per_subject=16,
        training=True,
        target_shape=(290, 244, 224),
    ):
        self.subject_ids = list(subject_ids)
        self.pred_dirs = pred_dirs
        self.gt_dir = gt_dir
        self.patch_size = tuple(patch_size)
        self.patches_per_subject = int(patches_per_subject)
        self.training = training
        self.target_shape = tuple(target_shape)
        self.vol_shape = self.target_shape

        if training:
            self.length = len(self.subject_ids) * self.patches_per_subject
        else:
            self.length = len(self.subject_ids)

    def __len__(self):
        return self.length

    def center_crop_or_pad(self, vol, target_shape):
        D, H, W = vol.shape
        tD, tH, tW = target_shape
        out = np.zeros(target_shape, dtype=vol.dtype)

        z0_src = max(0, (D - tD) // 2)
        y0_src = max(0, (H - tH) // 2)
        x0_src = max(0, (W - tW) // 2)

        z0_dst = max(0, (tD - D) // 2)
        y0_dst = max(0, (tH - H) // 2)
        x0_dst = max(0, (tW - W) // 2)

        z_len = min(D, tD)
        y_len = min(H, tH)
        x_len = min(W, tW)

        out[
            z0_dst:z0_dst + z_len,
            y0_dst:y0_dst + y_len,
            x0_dst:x0_dst + x_len
        ] = vol[
            z0_src:z0_src + z_len,
            y0_src:y0_src + y_len,
            x0_src:x0_src + x_len
        ]
        return out

    def load_subject(self, subj):
        x0 = np.load(os.path.join(self.pred_dirs[0], f"subject_{subj:04d}.npy"))
        x1 = np.load(os.path.join(self.pred_dirs[1], f"subject_{subj:04d}.npy"))
        x2 = np.load(os.path.join(self.pred_dirs[2], f"subject_{subj:04d}.npy"))
        y  = np.load(os.path.join(self.gt_dir,      f"subject_{subj:04d}.npy"))

        target_shape = self.target_shape

        if x0.shape != target_shape:
            x0 = self.center_crop_or_pad(x0, target_shape)
        if x1.shape != target_shape:
            x1 = self.center_crop_or_pad(x1, target_shape)
        if x2.shape != target_shape:
            x2 = self.center_crop_or_pad(x2, target_shape)
        if y.shape != target_shape:
            y = self.center_crop_or_pad(y, target_shape)

        x = np.stack([x0, x1, x2], axis=0)
        y = y[None, ...]
        return x.astype(np.float32), y.astype(np.float32)

    def random_patch_coords(self, shape, patch_size):
        D, H, W = shape
        pd, ph, pw = patch_size

        if pd > D or ph > H or pw > W:
            raise ValueError(
                f"patch_size {patch_size} larger than volume shape {shape}"
            )

        z = np.random.randint(0, D - pd + 1)
        y = np.random.randint(0, H - ph + 1)
        x = np.random.randint(0, W - pw + 1)
        return z, y, x

    def __getitem__(self, idx):
        if self.training:
            subj = self.subject_ids[idx // self.patches_per_subject]
            x, y = self.load_subject(subj)

            _, D, H, W = x.shape
            z, yy, xx = self.random_patch_coords((D, H, W), self.patch_size)
            pd, ph, pw = self.patch_size

            x_patch = x[:, z:z+pd, yy:yy+ph, xx:xx+pw]
            y_patch = y[:, z:z+pd, yy:yy+ph, xx:xx+pw]
            return torch.from_numpy(x_patch), torch.from_numpy(y_patch), subj
        else:
            subj = self.subject_ids[idx]
            x, y = self.load_subject(subj)
            return torch.from_numpy(x), torch.from_numpy(y), subj


# =========================================================
# ---------------- fusion model ---------------------------
# =========================================================

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FusionUNet3D(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base_ch=16):
        super().__init__()

        self.enc1 = ConvBlock3D(in_ch, base_ch)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock3D(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ConvBlock3D(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose3d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_ch * 2, base_ch)

        self.out = nn.Conv3d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


# =========================================================
# -------- sliding-window inference for fusion -----------
# =========================================================

def sliding_window_inference_3d(model, vol, patch_size, device):
    """
    vol: [1, C, D, H, W]
    returns [1, 1, D, H, W]
    """
    _, _, D, H, W = vol.shape
    pd, ph, pw = patch_size

    stride = (max(pd // 2, 1), max(ph // 2, 1), max(pw // 2, 1))

    out = torch.zeros((1, 1, D, H, W), dtype=torch.float32, device=device)
    cnt = torch.zeros((1, 1, D, H, W), dtype=torch.float32, device=device)

    z_list = list(range(0, max(D - pd, 0) + 1, stride[0]))
    y_list = list(range(0, max(H - ph, 0) + 1, stride[1]))
    x_list = list(range(0, max(W - pw, 0) + 1, stride[2]))

    if len(z_list) == 0 or z_list[-1] != max(D - pd, 0):
        z_list.append(max(D - pd, 0))
    if len(y_list) == 0 or y_list[-1] != max(H - ph, 0):
        y_list.append(max(H - ph, 0))
    if len(x_list) == 0 or x_list[-1] != max(W - pw, 0):
        x_list.append(max(W - pw, 0))

    model.eval()
    with torch.no_grad():
        for z in z_list:
            for y in y_list:
                for x in x_list:
                    patch = vol[:, :, z:z+pd, y:y+ph, x:x+pw].to(device)
                    pred = model(patch)
                    out[:, :, z:z+pd, y:y+ph, x:x+pw] += pred
                    cnt[:, :, z:z+pd, y:y+ph, x:x+pw] += 1.0

    out = out / torch.clamp(cnt, min=1.0)
    return out


# =========================================================
# ---------------- metrics -------------------------------
# =========================================================

def compute_metrics_single(y_true, y_pred):
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    data_range = 1.0

    psnr = peak_signal_noise_ratio(y_true, y_pred, data_range=data_range)
    ssim = structural_similarity(y_true, y_pred, data_range=data_range)

    f_true = y_true.reshape(-1)
    f_pred = y_pred.reshape(-1)
    mask = f_true > 0

    if np.any(mask):
        psnr_brain = peak_signal_noise_ratio(f_true[mask], f_pred[mask], data_range=data_range)
        ssim_brain = structural_similarity(f_true[mask], f_pred[mask], data_range=data_range)
    else:
        psnr_brain = np.nan
        ssim_brain = np.nan

    return {
        "PSNR": psnr,
        "SSIM": ssim,
        "PSNR_brain": psnr_brain,
        "SSIM_brain": ssim_brain,
    }


# =========================================================
# ---------------- fusion training ------------------------
# =========================================================

def train_fusion_model(
    out_dir,
    train_subjects,
    test_subjects=None,
    pred_dirs=None,
    gt_dir=None,
    ref_nifti_dir=None,
    ref_pattern="{subject}.nii.gz",
    save_nifti=True,
    fusion_epochs=40,
    fusion_lr=5e-5,
    fusion_batch_size=2,
    patch_size=(64, 64, 64),
    patches_per_subject=16,
    use_perceptual_loss=False,
    perceptual_weight=0.005,
    num_workers=4,
    device="cuda",
):
    """
    Train-only fusion model.

    This version does not use idx_test from the axis-specific checkpoints and
    does not run validation/testing inside this script. All subjects passed in
    train_subjects are used for training. Evaluation should be performed
    separately.
    """
    fusion_dir = os.path.join(out_dir, "fusion")
    mkdir(fusion_dir)

    fusion_target_shape = (290, 244, 224)

    train_ds = FusionPatchDataset(
        subject_ids=train_subjects,
        pred_dirs=pred_dirs,
        gt_dir=gt_dir,
        patch_size=patch_size,
        patches_per_subject=patches_per_subject,
        training=True,
        target_shape=fusion_target_shape,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=fusion_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = FusionUNet3D(in_ch=3, out_ch=1, base_ch=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=fusion_lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    l1_loss = nn.L1Loss()

    if use_perceptual_loss:
        perc_loss = PerceptualLoss(
            spatial_dims=3,
            is_fake_3d=True,
            network_type="radimagenet_resnet50"
        ).to(device)
    else:
        perc_loss = None

    train_l1_list = []
    train_perc_list = []

    for epoch in range(fusion_epochs):
        t0 = time.time()
        model.train()

        epoch_l1 = 0.0
        epoch_perc = 0.0
        n_step = 0

        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred = model(xb)
            loss_l1 = l1_loss(pred, yb)

            if use_perceptual_loss:
                loss_p = perc_loss(pred, yb)
            else:
                loss_p = torch.tensor(0.0, device=device)

            loss = loss_l1 + perceptual_weight * loss_p

            loss.backward()
            optimizer.step()

            epoch_l1 += loss_l1.item()
            epoch_perc += loss_p.item()
            n_step += 1

            del xb, yb, pred, loss, loss_l1, loss_p

        scheduler.step()

        epoch_l1 /= max(n_step, 1)
        epoch_perc /= max(n_step, 1)

        train_l1_list.append(epoch_l1)
        train_perc_list.append(epoch_perc)

        df_loss = pd.DataFrame({
            "train_l1": train_l1_list,
            "train_perc": train_perc_list,
        })
        df_loss.to_csv(os.path.join(fusion_dir, "losses.csv"), index=False)

        save_dict = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_l1": train_l1_list,
            "train_perc": train_perc_list,
            "patch_size": patch_size,
            "train_subjects": list(map(int, train_subjects)),
            "fusion_target_shape": fusion_target_shape,
        }

        torch.save(save_dict, os.path.join(fusion_dir, "fusion_last.pt"))

        print(
            f"[fusion] epoch {epoch+1}/{fusion_epochs} "
            f"train_l1={epoch_l1:.6f} train_perc={epoch_perc:.6f} "
            f"time={time.time()-t0:.1f}s",
            flush=True
        )

        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    plt.figure()
    plt.plot(range(1, len(train_l1_list)+1), train_l1_list, label="train_l1")
    plt.xlabel("epoch")
    plt.ylabel("L1 loss")
    plt.legend()
    plt.title("Fusion training L1")
    plt.savefig(os.path.join(fusion_dir, "fusion_l1.png"))
    plt.close()

    plt.figure()
    plt.plot(range(1, len(train_perc_list)+1), train_perc_list, label="train_perc")
    plt.xlabel("epoch")
    plt.ylabel("Perceptual loss")
    plt.legend()
    plt.title("Fusion training perceptual")
    plt.savefig(os.path.join(fusion_dir, "fusion_perc.png"))
    plt.close()


# =========================================================
# ---------------- main ----------------------------------
# =========================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    mkdir(args.out_dir)

    axis_cfgs = get_axis_cfgs(args)

    # ---------- check subject count ----------
    n_subjects_list = []
    axis_settings = {}

    for ax in [0, 1, 2]:
        st = load_settings(axis_cfgs[ax]["param"])
        axis_settings[ax] = st
        n_subjects_ax = infer_subject_count(axis_cfgs[ax]["inputs"], int(st["n_slices"]))
        if args.max_subjects is not None:
            n_subjects_ax = min(n_subjects_ax, args.max_subjects)
        n_subjects_list.append(n_subjects_ax)

    if len(set(n_subjects_list)) != 1:
        raise ValueError(f"Different subject counts across axes: {n_subjects_list}")

    n_subjects = n_subjects_list[0]
    print("n_subjects =", n_subjects)

    # ---------- run inference for three axes ----------
    axis_infos = {}

    for ax in [0, 1, 2]:
        info = run_axis_inference_for_all_subjects(
            axis_id=ax,
            axis_cfg=axis_cfgs[ax],
            out_dir=args.out_dir,
            device=device,
            infer_batch_size=args.infer_batch_size,
            max_subjects=args.max_subjects,
            ref_nifti_dir=args.ref_nifti_dir,
            ref_pattern=args.ref_nifti_pattern,
            save_nifti=args.save_nifti
        )
        axis_infos[ax] = info

    # ---------- training subjects ----------
    # This script is only used to train the fusion model.
    # Evaluation will be performed separately on another server.
    # Therefore, do not use idx_test stored in the axis-specific checkpoints.
    idx_train = np.arange(n_subjects, dtype=int)
    idx_test = np.array([], dtype=int)

    pd.DataFrame({"subject_idx": idx_train}).to_csv(
        os.path.join(args.out_dir, "fusion_train_subjects.csv"), index=False
    )
    pd.DataFrame({"subject_idx": idx_test}).to_csv(
        os.path.join(args.out_dir, "fusion_test_subjects.csv"), index=False
    )

    print("n_train =", len(idx_train))
    print("n_test =", len(idx_test))

    # ---------- canonical GT ----------
    canonical_gt_dir = axis_infos[args.canonical_axis]["gt_dir"]
    pred_dirs = {
        0: axis_infos[0]["pred_dir"],
        1: axis_infos[1]["pred_dir"],
        2: axis_infos[2]["pred_dir"],
    }

    # ---------- train fusion ----------
    train_fusion_model(
        out_dir=args.out_dir,
        train_subjects=idx_train,
        test_subjects=idx_test,
        pred_dirs=pred_dirs,
        gt_dir=canonical_gt_dir,
        ref_nifti_dir=args.ref_nifti_dir,
        ref_pattern=args.ref_nifti_pattern,
        save_nifti=args.save_nifti,
        fusion_epochs=args.fusion_epochs,
        fusion_lr=args.fusion_lr,
        fusion_batch_size=args.fusion_batch_size,
        patch_size=tuple(args.patch_size),
        patches_per_subject=args.patches_per_subject,
        use_perceptual_loss=args.fusion_use_perceptual_loss,
        perceptual_weight=args.fusion_perceptual_weight,
        num_workers=args.num_workers,
        device=device,
    )

    print("done fusion_CNN")


if __name__ == "__main__":
    main()
