#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("start LPIPS brain-mask 2D evaluation")

import argparse
import os
import ast

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import lpips


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


def get_slice(arr: np.ndarray, axis: int, idx: int):
    """
    Extract one 2D slice from a 3D volume along the given axis.
    """
    sl = [slice(None)] * arr.ndim
    sl[axis] = idx
    out = arr[tuple(sl)]
    return np.asarray(out)


def prepare_lpips_tensor_from_2d(im2d: np.ndarray, mask2d: np.ndarray, device: torch.device):
    """
    im2d: [H, W], assumed in [0,1]
    mask2d: [H, W], bool
    return tensor [1,3,H,W] in [-1,1]
    """
    masked = np.where(mask2d, im2d, 0.0).astype(np.float32)

    x = torch.from_numpy(masked).unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
    x = x.repeat(1, 3, 1, 1)                                 # [1,3,H,W]
    x = x * 2.0 - 1.0                                        # [0,1] -> [-1,1]
    x = x.to(device)
    return x


def compute_lpips_slicewise_brain_masked(
    im_gen_01: np.ndarray,
    im_true_01: np.ndarray,
    brain_mask: np.ndarray,
    slice_axis: int,
    lpips_model,
    device: torch.device
):
    """
    Slice-wise LPIPS with brain mask applied.
    For each 2D slice:
      - if mask is empty, skip
      - zero-out non-brain region in both images
      - compute LPIPS on masked slice
    Returns mean LPIPS over valid slices.
    """
    n_slices = im_true_01.shape[slice_axis]
    vals = []

    with torch.no_grad():
        for s in range(n_slices):
            gt2d = get_slice(im_true_01, slice_axis, s)
            gen2d = get_slice(im_gen_01, slice_axis, s)
            m2d = get_slice(brain_mask, slice_axis, s).astype(bool)

            if np.sum(m2d) == 0:
                continue

            t_gen = prepare_lpips_tensor_from_2d(gen2d, m2d, device)
            t_gt = prepare_lpips_tensor_from_2d(gt2d, m2d, device)

            val = lpips_model(t_gen, t_gt)
            vals.append(float(val.item()))

    if len(vals) == 0:
        return np.nan

    return float(np.nanmean(vals))


def main():
    ap = argparse.ArgumentParser("LPIPS brain-mask 2D evaluation from 3D nifti (metrics csv)")

    # Paths
    ap.add_argument("--big-path", type=str, required=True, help="Root path prefix (your big_path)")
    ap.add_argument("--info-csv", type=str, default="patient_info.csv", help="CSV filename or absolute path")
    ap.add_argument("--gen-dir", type=str, required=True, help="Generated nii directory (relative to big-path unless absolute)")
    ap.add_argument("--true-dir", type=str, required=True, help="Ground-truth nii directory (relative to big-path unless absolute)")
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

    # 2D evaluation axis
    ap.add_argument("--slice-axis", type=int, default=1, choices=[0, 1, 2],
                    help="Axis along which the 3D volume is sliced into 2D images for evaluation.")

    # Normalization knobs
    ap.add_argument("--clip-percentile", type=float, default=99.0,
                    help="Percentile used for clipping generated image (default 99)")

    # LPIPS
    ap.add_argument("--enable-lpips", action="store_true", help="Compute LPIPS")
    ap.add_argument("--lpips-net", type=str, default="alex", choices=["alex", "vgg", "squeeze"],
                    help="Backbone used by LPIPS")
    ap.add_argument("--use-gpu-lpips", action="store_true",
                    help="Move LPIPS model + tensors to CUDA if available")

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

    device = torch.device("cuda" if (args.use_gpu_lpips and torch.cuda.is_available()) else "cpu")

    lpips_model = None
    if args.enable_lpips:
        lpips_model = lpips.LPIPS(net=args.lpips_net).to(device)
        lpips_model.eval()

    df = {
        "lpips_brain_2d": [],
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

        gen_name = f"{i}_t1_generated.nii.gz"
        true_name = f"{i}_normalized.nii.gz"

        # allow external naming by replacing below if needed
        # but keep same style as your current bash passing gen-dir
        gen_candidates = [f for f in os.listdir(gen_dir) if f.startswith(f"{i}_") and f.endswith(".nii.gz")]
        if len(gen_candidates) == 1:
            gen_name = gen_candidates[0]

        gen_path = os.path.join(gen_dir, gen_name)
        true_path = os.path.join(true_dir, true_name)

        if (not os.path.exists(gen_path)) or (not os.path.exists(true_path)):
            continue

        im_gen = nib.load(gen_path).get_fdata()
        im_true = nib.load(true_path).get_fdata()

        if im_gen.shape != im_true.shape:
            print(f"[warning] shape mismatch for patient {i}: gen={im_gen.shape}, true={im_true.shape}, skipped.")
            continue

        # generated -> [0,1]
        im_gen_01 = normalize_generated_to_0_1(im_gen, clip_percentile=args.clip_percentile)

        # true assumed already normalized; clip once for safety
        im_true_01 = np.clip(im_true, 0.0, 1.0).astype(np.float32)

        brain_mask = (im_true > 0)

        if args.enable_lpips:
            lpips_b2d = compute_lpips_slicewise_brain_masked(
                im_gen_01=im_gen_01,
                im_true_01=im_true_01,
                brain_mask=brain_mask,
                slice_axis=args.slice_axis,
                lpips_model=lpips_model,
                device=device,
            )
        else:
            lpips_b2d = np.nan

        df["lpips_brain_2d"].append(float(lpips_b2d))
        df["patient"].append(int(i))

        evaluated += 1
        print(f"patient {i}: lpips_brain_2d = {lpips_b2d}")

    def safe_mean(x):
        arr = np.array(x, dtype=float)
        if arr.size == 0:
            return np.nan
        return float(np.nanmean(arr))

    df["lpips_brain_2d"].append(safe_mean(df["lpips_brain_2d"]))
    df["patient"].append(0)

    out_df = pd.DataFrame(df)
    out_df.to_csv(args.metrics_csv, index=False)

    print(f"Evaluated subjects: {evaluated}")
    print(f"Saved: {args.metrics_csv}")


if __name__ == "__main__":
    main()
