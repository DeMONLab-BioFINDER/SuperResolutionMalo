#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("start")

import argparse
import os
import ast

import numpy as np
import pandas as pd
import nibabel as nib
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from generative.losses import PerceptualLoss


def parse_int_list(s: str):
    """
    Parse "1,2,3" or "1 2 3" or JSON-like "[1,2,3]" into List[int].
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            v = ast.literal_eval(s)
            return [int(x) for x in v]
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Cannot parse list from: {s}") from e
    parts = s.replace(",", " ").split()
    try:
        return [int(x) for x in parts]
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Cannot parse int list from: {s}") from e


def normalize_generated_to_0_1(im: np.ndarray, clip_percentile: float = 99.0, eps: float = 1e-8):
    """
    Follow original logic:
      - compute pmax on im>0 then clip im to [0, pmax]
      - min/max on clipped volume then normalize to [0,1]
    """
    v = im.flatten()
    v = v[v > 0]
    if v.size == 0:
        return np.zeros_like(im, dtype=np.float32)

    pmax = np.percentile(v, clip_percentile)
    v3T = np.clip(im, 0, pmax)

    vmin = float(np.min(v3T))
    vmax = float(np.max(v3T))
    denom = max(vmax - vmin, eps)
    out = (im - vmin) / denom
    return out.astype(np.float32)


def compute_metrics(im_gen_01: np.ndarray, im_true: np.ndarray):
    """
    PSNR/SSIM on full volume, data_range=1
    """
    psnr = peak_signal_noise_ratio(im_true, im_gen_01, data_range=1)
    ssim = structural_similarity(im_true, im_gen_01, data_range=1)
    return psnr, ssim


def compute_metrics_masked(im_gen_01: np.ndarray, im_true: np.ndarray, mask: np.ndarray):
    """
    Masked PSNR/SSIM using flattened masked vectors; data_range=1
    """
    f_true = im_true.flatten()
    f_gen = im_gen_01.flatten()

    m = mask.flatten().astype(bool)
    f_true_m = f_true[m]
    f_gen_m = f_gen[m]

    if f_true_m.size == 0:
        return np.nan, np.nan

    psnr = peak_signal_noise_ratio(f_true_m, f_gen_m, data_range=1)
    ssim = structural_similarity(f_true_m, f_gen_m, data_range=1)
    return psnr, ssim


def slice_along_axis(arr: np.ndarray, axis: int, start: int):
    """
    Generic crop like arr[:, start:, :] but for arbitrary axis.
    """
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(start, None)
    return arr[tuple(sl)]

def parse_ranges_cell(cell):
    """
    Parse a CSV cell like '[[53,182],[270,315]]' into a Python list of ranges.
    Returns a list of [start, end] pairs.
    """
    if pd.isna(cell):
        return []

    if isinstance(cell, str):
        cell = cell.strip()
        if not cell:
            return []
        ranges = ast.literal_eval(cell)
    else:
        ranges = cell

    out = []
    for r in ranges:
        if len(r) != 2:
            continue
        start, end = int(r[0]), int(r[1])
        out.append([start, end])
    return out

def parse_ranges_cell(cell):
    """
    Parse a CSV cell like '[[53,182],[270,315]]' into a Python list of ranges.
    Returns a list of [start, end] pairs.
    """
    if pd.isna(cell):
        return []

    if isinstance(cell, str):
        cell = cell.strip()
        if not cell:
            return []
        ranges = ast.literal_eval(cell)
    else:
        ranges = cell

    out = []
    for r in ranges:
        if len(r) != 2:
            continue
        start, end = int(r[0]), int(r[1])
        out.append([start, end])
    return out

def build_axis_range_mask(shape, axis, ranges):
    """
    Build a boolean mask of shape `shape`.
    Keeps voxels whose slice index along `axis` falls inside any interval in `ranges`.

    Each range is interpreted as [start, end], inclusive.
    """
    mask = np.zeros(shape, dtype=bool)

    for start, end in ranges:
        sl = [slice(None)] * len(shape)
        sl[axis] = slice(start, end + 1)
        mask[tuple(sl)] = True

    return mask


def main():
    ap = argparse.ArgumentParser("GAN 3D evaluation (metrics csv)")

    # Paths
    ap.add_argument("--big-path", type=str, required=True, help="Root path prefix (your big_path)")
    ap.add_argument("--info-csv", type=str, default="patient_info2.csv", help="CSV filename or absolute path")
    ap.add_argument("--gen-dir", type=str, default="to_seg/Gan_seg", help="Generated nii directory (relative to big-path unless absolute)")
    ap.add_argument("--gen-suffix", type=str, default="_t1_generated.nii.gz", help="Generated filename suffix")
    ap.add_argument("--true-dir", type=str, default="Data_norm/7T", help="Ground-truth nii directory (relative to big-path unless absolute)")
    ap.add_argument("--true-suffix", type=str, default="_normalized.nii.gz", help="GT filename suffix")
    ap.add_argument("--metrics-csv", type=str, required=True, help="Output CSV path")

    # Dataset indexing / filtering
    ap.add_argument("--idx-prefix", type=str, default="7T_015_BioFINDER_", help="Prefix used in info[7T_idx]")
    ap.add_argument("--idx-col", type=str, default="7T_idx", help="Column name for idx match in info CSV")
    ap.add_argument("--usage-col", type=str, default="usage", help="Column name for usage in info CSV")
    ap.add_argument("--usage-value", type=str, default="test", help="Which usage to evaluate (default=test)")
    ap.add_argument("--start-idx", type=int, default=126)
    ap.add_argument("--end-idx", type=int, default=304)
    ap.add_argument("--missing", type=parse_int_list, default=None,
                    help='Missing indices list, e.g. "203,204,223" or "[203,204]"')

    # Corrupt slice handling
    ap.add_argument("--eval-axis", type=int, default=1, choices=[0, 1, 2],
                    help="Axis used for no_corrupt evaluation cropping (0/1/2). Independent from input/model axis.")
    ap.add_argument("--disable-no-corrupt", action="store_true", help="Do not compute no_corrupt metrics")

    # Normalization knobs
    ap.add_argument("--clip-percentile", type=float, default=99.0, help="Percentile used for clipping generated image (default 99)")

    # Perceptual loss
    ap.add_argument("--enable-perceptual", action="store_true", help="Compute perceptual losses")
    ap.add_argument("--perceptual-net10", type=str, default="medicalnet_resnet10_23datasets")
    ap.add_argument("--perceptual-net50", type=str, default="medicalnet_resnet50_23datasets")
    ap.add_argument("--use-gpu-perceptual", action="store_true", help="Move perceptual loss models + tensors to CUDA if available")

    # Logging
    ap.add_argument("--print-ratio", action="store_true", help="Print len(masked_gen)/len(all) ratio per subject (your original print)")

    # pattern for input and output
    ap.add_argument("--gen-pattern", type=str, default="{i}_t1_generated.nii.gz",
                    help='Filename pattern under gen-dir. Use "{i}" as placeholder.')
    ap.add_argument("--true-pattern", type=str, default="{i}_normalized.nii.gz",
                    help='Filename pattern under true-dir. Use "{i}" as placeholder.')

    args = ap.parse_args()

    big_path = args.big_path
    if not big_path.endswith("/"):
        big_path += "/"

    info_path = args.info_csv
    if not os.path.isabs(info_path):
        info_path = os.path.join(big_path, info_path)

    gen_dir = args.gen_dir
    if not os.path.isabs(gen_dir):
        gen_dir = os.path.join(big_path, gen_dir)

    true_dir = args.true_dir
    if not os.path.isabs(true_dir):
        true_dir = os.path.join(big_path, true_dir)

    os.makedirs(os.path.dirname(args.metrics_csv), exist_ok=True)

    info = pd.read_csv(info_path)

    if args.missing is None:
        args.missing = [203, 204, 223, 248, 274, 288, 300, 184, 192]

    device = torch.device("cuda" if (args.use_gpu_perceptual and torch.cuda.is_available()) else "cpu")

    model10 = None
    model50 = None
    if args.enable_perceptual:
        model10 = PerceptualLoss(spatial_dims=3, is_fake_3d=False, network_type=args.perceptual_net10).to(device)
        model50 = PerceptualLoss(spatial_dims=3, is_fake_3d=False, network_type=args.perceptual_net50).to(device)
        model10.eval()
        model50.eval()

    df = {
        "psnr": [], "ssim": [],
        "psnr_brain": [], "ssim_brain": [],
        "psnr_no_corrupt": [], "ssim_no_corrupt": [],
        "perc_10": [], "perc_50": [],
        "patient": []
    }

    evaluated = 0

    for i in range(args.start_idx, args.end_idx + 1):
        if i in args.missing:
            continue

        key = f"{args.idx_prefix}{i}"
        info_i = info.loc[info[args.idx_col] == key]

        if info_i.shape[0] == 0:
            continue

        if str(info_i[args.usage_col].iloc[0]) != args.usage_value:
            continue

        gen_name = args.gen_pattern.format(i=i)
        true_name = args.true_pattern.format(i=i)

        gen_path = os.path.join(gen_dir, gen_name)
        true_path = os.path.join(true_dir, true_name)

        if (not os.path.exists(gen_path)) or (not os.path.exists(true_path)):
            continue

        im_gen = nib.load(gen_path).get_fdata()
        im_true = nib.load(true_path).get_fdata()

        im_gen_01 = normalize_generated_to_0_1(im_gen, clip_percentile=args.clip_percentile)

        if args.enable_perceptual:
            with torch.no_grad():
                t_gen = torch.from_numpy(im_gen_01)[None, None, ...].float().to(device)
                t_true = torch.from_numpy(im_true)[None, None, ...].float().to(device)
                df["perc_10"].append(float(model10(t_gen, t_true).item()))
                df["perc_50"].append(float(model50(t_gen, t_true).item()))
        else:
            df["perc_10"].append(np.nan)
            df["perc_50"].append(np.nan)

        psnr, ssim = compute_metrics(im_gen_01, im_true)
        df["psnr"].append(float(psnr))
        df["ssim"].append(float(ssim))
        df["patient"].append(int(i))

        mask_brain = (im_true > 0)
        f_true = im_true.flatten()

        if args.print_ratio:
            ratio = float(np.sum(mask_brain.flatten()) / f_true.size) if f_true.size > 0 else np.nan
            print(ratio)

        psnr_b, ssim_b = compute_metrics_masked(im_gen_01, im_true, mask_brain)
        df["psnr_brain"].append(float(psnr_b))
        df["ssim_brain"].append(float(ssim_b))



        # no_corrupt metrics
        if args.disable_no_corrupt:
            df["psnr_no_corrupt"].append(np.nan)
            df["ssim_no_corrupt"].append(np.nan)
        else:
            # 1) read valid non-corrupt ranges from CSV
            valid_col = f"valid_ranges_{args.eval_axis}"
            valid_ranges = parse_ranges_cell(info_i[valid_col].iloc[0])

            # 2) read brain-valid range from bot_x / top_x
            bot_col = f"bot_{args.eval_axis}"
            top_col = f"top_{args.eval_axis}"
            bot = int(info_i[bot_col].iloc[0])
            top = int(info_i[top_col].iloc[0])

            # 3) build mask for non-corrupt valid ranges
            mask_valid_ranges = build_axis_range_mask(im_true.shape, args.eval_axis, valid_ranges)

            # 4) build mask for brain-valid span along the same axis
            mask_brain_span = build_axis_range_mask(im_true.shape, args.eval_axis, [[bot, top]])
        
            # 5) keep only voxels that are:
            #    - inside valid non-corrupt ranges
            #    - inside brain-valid span
            #    - inside true brain mask (im_true > 0)
            mask_s = mask_valid_ranges & mask_brain_span & (im_true > 0)

            psnr_nc, ssim_nc = compute_metrics_masked(im_gen_01, im_true, mask_s)
            df["psnr_no_corrupt"].append(float(psnr_nc))
            df["ssim_no_corrupt"].append(float(ssim_nc))
        evaluated += 1


    def safe_mean(x):
        arr = np.array(x, dtype=float)
        if arr.size == 0:
            return np.nan
        return float(np.nanmean(arr))

    df["psnr"].append(safe_mean(df["psnr"]))
    df["ssim"].append(safe_mean(df["ssim"]))
    df["psnr_brain"].append(safe_mean(df["psnr_brain"]))
    df["ssim_brain"].append(safe_mean(df["ssim_brain"]))
    df["perc_10"].append(safe_mean(df["perc_10"]))
    df["perc_50"].append(safe_mean(df["perc_50"]))
    df["psnr_no_corrupt"].append(safe_mean(df["psnr_no_corrupt"]))
    df["ssim_no_corrupt"].append(safe_mean(df["ssim_no_corrupt"]))
    df["patient"].append(0)

    out_df = pd.DataFrame(df)
    out_df.to_csv(args.metrics_csv, index=False)

    print(f"Evaluated subjects: {evaluated}")
    print(f"Saved: {args.metrics_csv}")


if __name__ == "__main__":
    main()
