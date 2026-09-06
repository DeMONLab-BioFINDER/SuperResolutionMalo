#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("start inferer_CNN")

import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from generative.losses import PerceptualLoss
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# =========================================================
# ---------------------- argparse -------------------------
# =========================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="3-axis nifti -> Fusion CNN inference + 7T evaluation"
    )

    p.add_argument(
        "--big-path",
        type=str,
        default="/proj/nobackup/sens2023026/wharf/rzshen/rzshen-sens2023026/",
        help="Project root"
    )

    # input nifti dirs for three axes
    p.add_argument("--axis0-dir", type=str, required=True)
    p.add_argument("--axis1-dir", type=str, required=True)
    p.add_argument("--axis2-dir", type=str, required=True)

    # nifti filename patterns
    p.add_argument(
        "--axis0-pattern",
        type=str,
        default="{patient}_t1_generated__axis0_on_UnetModel0.nii.gz"
    )
    p.add_argument(
        "--axis1-pattern",
        type=str,
        default="{patient}_t1_generated__axis1_on_UnetModel1.nii.gz"
    )
    p.add_argument(
        "--axis2-pattern",
        type=str,
        default="{patient}_t1_generated__axis2_on_UnetModel2.nii.gz"
    )

    # fusion checkpoint
    p.add_argument("--fusion-ckpt", type=str, required=True)

    # output
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--metrics-csv", type=str, required=True)

    # data / patient info
    p.add_argument("--patient-csv", type=str, default="patient_info2.csv")
    p.add_argument("--start-idx", type=int, default=126)
    p.add_argument("--end-idx", type=int, default=304)

    p.add_argument("--parent-7t-norm", type=str, default=None,
                   help="Directory for normalized 7T volumes, e.g. <big_path>/Data_norm/7T")

    # inference
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--patch-size", type=int, nargs=3, default=None,
                   help="If omitted, use patch_size saved in fusion checkpoint.")
    p.add_argument("--base-ch", type=int, default=16)

    # evaluation
    p.add_argument("--eval-axis", type=int, default=0, choices=[0, 1, 2],
                   help="Axis used for no_corrupt metric. Usually set to canonical axis used in fusion training.")
    p.add_argument("--use-gpu-perceptual", action="store_true")

    # optional clipping
    p.add_argument("--clip-min", type=float, default=None)
    p.add_argument("--clip-max", type=float, default=None)

    return p.parse_args()


# =========================================================
# ---------------------- model ----------------------------
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
# ---------------------- helpers --------------------------
# =========================================================

def safe_torch_load(path, map_location="cpu"):
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    return ckpt


def strip_module_prefix(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd


def center_crop_or_pad(vol, target_shape):
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


def slice_along_axis(arr: np.ndarray, axis: int, start: int):
    sl = [slice(None)] * 3
    sl[axis] = slice(start, None)
    return arr[tuple(sl)]


def get_no_corrupt_shift(info_df: pd.DataFrame, key: str, axis: int):
    if f"bot_{axis}" in info_df.columns:
        return int(info_df.loc[info_df["7T_idx"] == key, f"bot_{axis}"].iloc[0])
    elif "bot_first_slice" in info_df.columns:
        return int(info_df.loc[info_df["7T_idx"] == key, "bot_first_slice"].iloc[0])
    else:
        return 0


def maybe_clip(arr, clip_min=None, clip_max=None):
    if clip_min is None and clip_max is None:
        return arr
    return np.clip(arr, clip_min, clip_max)


def load_nifti_data(path):
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    return img, data


def sliding_window_inference_3d(model, vol, patch_size, device):
    """
    vol: [1, C, D, H, W]
    return: [1, 1, D, H, W]
    """
    _, _, D, H, W = vol.shape
    pd, ph, pw = patch_size

    if pd > D or ph > H or pw > W:
        raise ValueError(
            f"patch_size {patch_size} is larger than input volume shape {(D, H, W)}"
        )

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
# ---------------------- metrics --------------------------
# =========================================================

def compute_psnr_ssim_full(y_true, y_pred, data_range=1.0):
    psnr = peak_signal_noise_ratio(y_true, y_pred, data_range=data_range)
    ssim = structural_similarity(y_true, y_pred, data_range=data_range)
    return psnr, ssim


def compute_psnr_ssim_brain(y_true, y_pred, data_range=1.0):
    f_true = y_true.flatten()
    f_pred = y_pred.flatten()

    mask = f_true > 0
    if np.sum(mask) == 0:
        return np.nan, np.nan

    f_true = f_true[mask]
    f_pred = f_pred[mask]

    psnr = peak_signal_noise_ratio(f_true, f_pred, data_range=data_range)
    ssim = structural_similarity(f_true, f_pred, data_range=data_range)
    return psnr, ssim


def compute_psnr_ssim_no_corrupt(y_true, y_pred, axis, shifting, data_range=1.0):
    y_true_s = slice_along_axis(y_true, axis, shifting)
    y_pred_s = slice_along_axis(y_pred, axis, shifting)

    f_true = y_true_s.flatten()
    f_pred = y_pred_s.flatten()

    mask = f_true > 0
    if np.sum(mask) == 0:
        return np.nan, np.nan

    f_true = f_true[mask]
    f_pred = f_pred[mask]

    psnr = peak_signal_noise_ratio(f_true, f_pred, data_range=data_range)
    ssim = structural_similarity(f_true, f_pred, data_range=data_range)
    return psnr, ssim


# =========================================================
# ---------------------- main -----------------------------
# =========================================================

def main():
    args = parse_args()
    t_start = time.time()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    big_path = Path(args.big_path)

    axis0_dir = Path(args.axis0_dir)
    axis1_dir = Path(args.axis1_dir)
    axis2_dir = Path(args.axis2_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_csv).parent.mkdir(parents=True, exist_ok=True)

    if args.parent_7t_norm is None:
        parent_7t_norm = big_path / "Data_norm/7T"
    else:
        parent_7t_norm = Path(args.parent_7t_norm)

    info = pd.read_csv(big_path / args.patient_csv)

    # keep same exclusion as your old script
    i_missing = [203, 204, 223, 248, 274, 288, 300, 184, 192]
    idx_7T_prefix = "7T_015_BioFINDER_"

    # load fusion checkpoint
    ckpt = safe_torch_load(args.fusion_ckpt, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    state_dict = strip_module_prefix(state_dict)

    patch_size = tuple(args.patch_size) if args.patch_size is not None else tuple(ckpt.get("patch_size", [64, 64, 64]))
    print("patch_size =", patch_size)

    model = FusionUNet3D(in_ch=3, out_ch=1, base_ch=args.base_ch)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # perceptual metrics models
    model10 = PerceptualLoss(
        spatial_dims=3,
        is_fake_3d=False,
        network_type="medicalnet_resnet10_23datasets",
    )
    model50 = PerceptualLoss(
        spatial_dims=3,
        is_fake_3d=False,
        network_type="medicalnet_resnet50_23datasets",
    )

    if args.use_gpu_perceptual:
        model10 = model10.to(device)
        model50 = model50.to(device)

    df = {
        "psnr": [],
        "ssim": [],
        "psnr_brain": [],
        "ssim_brain": [],
        "psnr_no_corrupt": [],
        "ssim_no_corrupt": [],
        "perc_10": [],
        "perc_50": [],
        "patient": [],
    }

    for i in range(args.start_idx, args.end_idx + 1):
        key = idx_7T_prefix + str(i)
        info_i = info.loc[info["7T_idx"] == key]

        if len(info_i) == 0:
            continue
        if i in i_missing:
            continue
        if info_i["usage"].iloc[0] != "test":
            continue

        print(f"\n=== Patient {i} ===")

        path0 = axis0_dir / args.axis0_pattern.format(patient=i)
        path1 = axis1_dir / args.axis1_pattern.format(patient=i)
        path2 = axis2_dir / args.axis2_pattern.format(patient=i)
        path_gt = parent_7t_norm / f"{i}_normalized.nii.gz"

        if not path0.exists():
            print(f"skip {i}: missing {path0}")
            continue
        if not path1.exists():
            print(f"skip {i}: missing {path1}")
            continue
        if not path2.exists():
            print(f"skip {i}: missing {path2}")
            continue
        if not path_gt.exists():
            print(f"skip {i}: missing {path_gt}")
            continue

        img0, vol0 = load_nifti_data(path0)
        _, vol1 = load_nifti_data(path1)
        _, vol2 = load_nifti_data(path2)
        _, gt = load_nifti_data(path_gt)

        # match shapes to GT
        if vol0.shape != gt.shape:
            print(f"patient {i}: axis0 shape {vol0.shape} -> center_crop_or_pad -> {gt.shape}")
            vol0 = center_crop_or_pad(vol0, gt.shape)
        if vol1.shape != gt.shape:
            print(f"patient {i}: axis1 shape {vol1.shape} -> center_crop_or_pad -> {gt.shape}")
            vol1 = center_crop_or_pad(vol1, gt.shape)
        if vol2.shape != gt.shape:
            print(f"patient {i}: axis2 shape {vol2.shape} -> center_crop_or_pad -> {gt.shape}")
            vol2 = center_crop_or_pad(vol2, gt.shape)

        x = np.stack([vol0, vol1, vol2], axis=0).astype(np.float32)
        x_t = torch.from_numpy(x)[None, ...].float()

        with torch.no_grad():
            pred = sliding_window_inference_3d(model, x_t, patch_size, device)
        pred_np = pred[0, 0].detach().cpu().numpy().astype(np.float32)

        pred_np = maybe_clip(pred_np, args.clip_min, args.clip_max)

        # perceptual metrics
        if args.use_gpu_perceptual:
            pred_t = torch.from_numpy(pred_np)[None, None].float().to(device)
            gt_t = torch.from_numpy(gt)[None, None].float().to(device)
        else:
            pred_t = torch.from_numpy(pred_np)[None, None].float()
            gt_t = torch.from_numpy(gt)[None, None].float()

        perc10 = model10(pred_t, gt_t).item()
        perc50 = model50(pred_t, gt_t).item()

        # PSNR/SSIM
        psnr, ssim = compute_psnr_ssim_full(gt, pred_np, data_range=1.0)
        psnr_brain, ssim_brain = compute_psnr_ssim_brain(gt, pred_np, data_range=1.0)

        shifting = get_no_corrupt_shift(info, key, args.eval_axis)
        psnr_nc, ssim_nc = compute_psnr_ssim_no_corrupt(
            gt, pred_np, axis=args.eval_axis, shifting=shifting, data_range=1.0
        )

        # save nifti
        out_path = output_dir / f"{i}_t1_generated__fusionCNN.nii.gz"
        nib.save(nib.Nifti1Image(pred_np, img0.affine, header=img0.header), str(out_path))
        print(f"Saved: {out_path}")

        print(
            f"PSNR={psnr:.4f} SSIM={ssim:.4f} "
            f"PSNR_brain={psnr_brain:.4f} SSIM_brain={ssim_brain:.4f} "
            f"PSNR_no_corrupt={psnr_nc:.4f} SSIM_no_corrupt={ssim_nc:.4f} "
            f"perc10={perc10:.6f} perc50={perc50:.6f}"
        )

        df["psnr"].append(psnr)
        df["ssim"].append(ssim)
        df["psnr_brain"].append(psnr_brain)
        df["ssim_brain"].append(ssim_brain)
        df["psnr_no_corrupt"].append(psnr_nc)
        df["ssim_no_corrupt"].append(ssim_nc)
        df["perc_10"].append(perc10)
        df["perc_50"].append(perc50)
        df["patient"].append(i)

    # mean row
    if len(df["patient"]) > 0:
        df["psnr"].append(float(np.mean(np.array(df["psnr"]))))
        df["ssim"].append(float(np.mean(np.array(df["ssim"]))))
        df["psnr_brain"].append(float(np.mean(np.array(df["psnr_brain"]))))
        df["ssim_brain"].append(float(np.mean(np.array(df["ssim_brain"]))))
        df["psnr_no_corrupt"].append(float(np.mean(np.array(df["psnr_no_corrupt"]))))
        df["ssim_no_corrupt"].append(float(np.mean(np.array(df["ssim_no_corrupt"]))))
        df["perc_10"].append(float(np.mean(np.array(df["perc_10"]))))
        df["perc_50"].append(float(np.mean(np.array(df["perc_50"]))))
        df["patient"].append(0)

    df_pd = pd.DataFrame(data=df)
    df_pd.to_csv(args.metrics_csv, index=False)
    print(f"\nMetrics saved to: {args.metrics_csv}")
    print(f"done inferer_CNN, total time = {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
