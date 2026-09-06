#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("start")

import argparse
import ast
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from generative.losses import PerceptualLoss
from medicalnet_models.models.resnet import (
    medicalnet_resnet10_23datasets,
    medicalnet_resnet50_23datasets,
)
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import nn

from WarvitoCodes.models_unet_v2_conditioned_AdaDM_2D import UNetModel


# -----------------------------
# Helpers
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="UNet inference + evaluation for 3D MRI (supports axis 0/1/2 slicing)."
    )
    p.add_argument("--axis", type=int, default=1, choices=[0, 1, 2],
                   help="Slicing axis in original volume (0/1/2). Default=1 (old behavior).")
    p.add_argument("--big-path", type=str,
                   default="/proj/nobackup/sens2023026/wharf/rzshen/rzshen-sens2023026/")
    p.add_argument("--path-model", type=str, default="models/my_unet_no_diag.pt")
    p.add_argument("--path-settings", type=str, default="models/params_no_diag.txt")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory to save generated nifti. If None, auto uses to_seg/UNet_no_diag_axis{axis}/")
    p.add_argument("--metrics-csv", type=str, default=None,
                   help="Path to save metrics csv. If None, auto uses to_seg/metricsUnet_n_diag_axis{axis}.csv")
    p.add_argument("--patient-csv", type=str, default="patient_info2.csv")
    p.add_argument("--start-idx", type=int, default=126)
    p.add_argument("--end-idx", type=int, default=304)
    p.add_argument("--use-gpu-perceptual", action="store_true",
                   help="If set, move perceptual-loss models + tensors to GPU (faster, more VRAM).")
    p.add_argument("--quiet-slices", action="store_true",
                   help="If set, do not print per-slice debug info.")
    p.add_argument("--parent-3t-norm", type=str, default=None,
               help="Directory for normalized 3T volumes, e.g. <big_path>/Data_norm/3T")
    p.add_argument("--parent-7t-norm", type=str, default=None,
               help="Directory for normalized 7T volumes, e.g. <big_path>/Data_norm/7T")
    p.add_argument("--model-dim", type=int, default=0,
               help="Model id tag used for output naming (e.g., UnetModel0/1/2...).")
    
    return p.parse_args()


def make_perm(axis: int):
    """Return permutation that moves `axis` to the front."""
    return [axis] + [i for i in range(3) if i != axis]


def invert_perm(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def crop_hw_if_needed(res_2d: torch.Tensor, h_target: int, w_target: int) -> torch.Tensor:
    """
    Center-crop a 2D tensor [H,W] if output size is larger than target.
    (Usually not needed in your current setup, but keeps code robust.)
    """
    h, w = int(res_2d.shape[0]), int(res_2d.shape[1])
    if h == h_target and w == w_target:
        return res_2d

    dh = max(0, h - h_target)
    dw = max(0, w - w_target)

    h0 = dh // 2
    w0 = dw // 2
    return res_2d[h0:h0 + h_target, w0:w0 + w_target]


def slice_along_axis(arr: np.ndarray, axis: int, start: int):
    """Equivalent to arr[:, start:, :] style but generic axis."""
    sl = [slice(None)] * 3
    sl[axis] = slice(start, None)
    return arr[tuple(sl)]


def get_bot_top_for_axis(info_df: pd.DataFrame, key: str, axis: int):
    """
    Read bot_{axis}, top_{axis}. Falls back if columns are missing.
    """
    bot_col = f"bot_{axis}"
    top_col = f"top_{axis}"

    if bot_col in info_df.columns and top_col in info_df.columns:
        bot = float(info_df.loc[info_df["7T_idx"] == key, bot_col].iloc[0])
        top = float(info_df.loc[info_df["7T_idx"] == key, top_col].iloc[0])
        return bot, top

    # fallback (old behavior)
    bot = float(info_df.loc[info_df["7T_idx"] == key, "bot_1"].iloc[0])
    top = float(info_df.loc[info_df["7T_idx"] == key, "top_1"].iloc[0])
    return bot, top


def build_context_from_info(info_df: pd.DataFrame, key: str, conditions, j: int, mid: float, n_slices: int):
    """
    Build context tensor with shape [1, len(conditions), 1] following EXACT order in `conditions`.
    For senior model: conditions=['Sex','Age','position'] and position=2*(j-mid)/n_slices
    """
    row = info_df.loc[info_df["7T_idx"] == key].iloc[0]
    ctx = torch.zeros(1, len(conditions), 1, dtype=torch.float32)

    for k, name in enumerate(conditions):
        nm = str(name).strip().lower()
        if nm in ("sex", "gender"):
            ctx[0, k, 0] = float(row["gender_baseline_variable"])
        elif nm == "age":
            ctx[0, k, 0] = float(row["age"])
        elif nm == "position":
            ctx[0, k, 0] = float(2.0 * (j - mid) / n_slices)
        else:
            ctx[0, k, 0] = 0.0

    return ctx


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    tag = f"axis{args.axis}_on_UnetModel{args.model_dim}"

    big_path = Path(args.big_path)
    # Normalized data directories (can be overridden by CLI)
    if args.parent_3t_norm is None:
        parent_3t_norm = big_path / "Data_norm/3T"
    else:
        parent_3t_norm = Path(args.parent_3t_norm)

    if args.parent_7t_norm is None:
        parent_7t_norm = big_path / "Data_norm/7T"
    else:
        parent_7t_norm = Path(args.parent_7t_norm)
    
    
    

    path_model = Path(args.path_model)
    path_settings = Path(args.path_settings)

    #if args.output_dir is None:
    #    generated_7T = big_path / f"to_seg/UNet_no_diag_axis{args.axis}"
    #else:
    #    generated_7T = Path(args.output_dir)

    #if args.metrics_csv is None:
    #    metrics_csv_path = big_path / f"to_seg/metricsUnet_n_diag_axis{args.axis}.csv"
    #else:
    #    metrics_csv_path = Path(args.metrics_csv)
    if args.output_dir is None:
        generated_7T = big_path / f"to_seg/{tag}"
    else:
        generated_7T = Path(args.output_dir)

    if args.metrics_csv is None:
        metrics_csv_path = big_path / f"to_seg/metrics_{tag}.csv"
    else:
        metrics_csv_path = Path(args.metrics_csv)



    generated_7T.mkdir(parents=True, exist_ok=True)
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Load settings
    with open(path_settings, "r") as f:
        a = f.read().replace("\'", "\"")
        settings = ast.literal_eval(a)
    print(settings)

    # Data paths
    parent_3T = big_path / "jake__20240405_172444___t1___brain___fs/processed_reg/"
    parent_7T = big_path / "T1_7T_processed/"  # kept for compatibility / reference
    info = pd.read_csv(big_path / args.patient_csv)

    n_slices = settings["n_slices"]
    n_colors = settings["n_in_slices"]
    conditions = settings.get("conditions", ["Sex", "Age"])

    checkpoint = torch.load(path_model)

    # Number of downsamples
    n_down = len(settings["channel_mult"]) - 1 + int(settings["use_initial_down"])

    # Build model
    my_unet = UNetModel(
        image_size=settings["slice_size"],
        in_channels=n_colors,
        out_channels=1,
        model_channels=settings["n_channels"],
        num_res_blocks=settings["n_res_block"],
        attention_resolutions=settings["attention_res"],
        context_dim=settings["c_dim"],
        dropout=0.0,
        use_ada=settings["use_ada"],
        channel_mult=settings["channel_mult"],
        use_spatial_transformer=settings["use_spatial_transformer"],
        use_mid_attention=settings["use_mid_attention"],
        use_initial_downsample=settings["use_initial_down"],
        num_groups=settings["num_groups"],
    )

    my_unet = nn.DataParallel(my_unet)
    my_unet.load_state_dict(checkpoint["model_state_dict"])
    my_unet.cuda()
    my_unet.eval()

    # Perceptual metrics models (create once)
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
        model10 = model10.cuda()
        model50 = model50.cuda()

    # Metrics accumulators
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

    i_missing = [203, 204, 223, 248, 274, 288, 300, 184, 192]
    idx_7T_prefix = "7T_015_BioFINDER_"

    # Fixed timesteps tensor for diffusion-style UNet interface
    tps = torch.ones(1).cuda()

    # Main loop
    for i in range(args.start_idx, args.end_idx + 1):
        key = idx_7T_prefix + str(i)
        info_i = info.loc[info["7T_idx"] == key]

        if len(info_i) == 0:
            continue
        if i in i_missing:
            continue
        if info_i["usage"].iloc[0] != "test":
            continue

        print(f"\n=== Patient {i} | axis={args.axis} ===")

        # Position conditioning center uses selected axis
        bot_axis, top_axis = get_bot_top_for_axis(info, key, args.axis)
        mid = (bot_axis + top_axis) / 2.0

        # Load images
        # (In this script actual inference uses Data_norm/3T and Data_norm/7T)
        #im_nii = nib.load(str(big_path / f"Data_norm/3T/{i}_normalized.nii.gz"))
        #im2 = nib.load(str(big_path / f"Data_norm/7T/{i}_normalized.nii.gz")).get_fdata()
        im_nii = nib.load(str(parent_3t_norm / f"{i}_normalized.nii.gz"))
        im2 = nib.load(str(parent_7t_norm / f"{i}_normalized.nii.gz")).get_fdata()


        hd = im_nii.header
        af = im_nii.affine
        im = im_nii.get_fdata()

        # Reorder to [slice, H, W]
        perm = make_perm(args.axis)
        inv_perm = invert_perm(perm)
        im_sf = np.transpose(im, perm)  # [D, H, W]
        D, H, W = im_sf.shape
        print(f"slice-first shape: D={D}, H={H}, W={W} (perm={perm})")

        # Padding on H/W so UNet downsampling works
        if H % (2 ** n_down):
            r1 = (2 ** n_down) - (H % (2 ** n_down))
        else:
            r1 = 0
        if W % (2 ** n_down):
            r2 = (2 ** n_down) - (W % (2 ** n_down))
        else:
            r2 = 0

        print(f"pad H={r1}, W={r2}")

        x = torch.from_numpy(im_sf)
        del im, im_sf

        padder_hw = (r2 // 2, r2 // 2 + r2 % 2, r1 // 2, r1 // 2 + r1 % 2)
        x = F.pad(x, padder_hw, "constant", 0)  # pads last 2 dims of [D,H,W]

        # Conditions
        age = float(info.loc[info["7T_idx"] == key, "age"].iloc[0])
        sex = float(info.loc[info["7T_idx"] == key, "gender_baseline_variable"].iloc[0])

        # Output in slice-first orientation [D, H, W]
        ress_sf = torch.zeros(D, H, W)

        contexts_i = torch.ones(1, 2, 1)######################################
        
        
        
        contexts_i[0, 0, 0] = age
        contexts_i[0, 1, 0] = sex

        # Inference loop (single-slice stepping, like original code)
        with torch.no_grad():
            for j in range(n_colors // 2, D - (n_colors // 2)):
                #contexts_i[0, 2, 0] = 2 * (j - mid) / n_slices
                contexts_i = build_context_from_info(info, key, conditions, j=j, mid=mid, n_slices=n_slices)
                contexts_gpu = contexts_i.cuda()

                # x_i: [1, n_colors, H_pad, W_pad]
                x_i = x[None, j - n_colors // 2:j + n_colors // 2 + 1].float().cuda()

                if not args.quiet_slices:
                    print("context:", contexts_gpu)
                    print("x_i shape:", tuple(x_i.shape))

                res_i = my_unet(x_i, timesteps=tps, context=contexts_gpu)  # [1,1,H_pad,W_pad]
                res_2d = res_i[0, 0].detach().cpu()

                # Crop back to original H,W if pad caused larger output
                res_2d = crop_hw_if_needed(res_2d, H, W)

                # Fill back
                ress_sf[j] = res_2d

        # Convert slice-first output back to original volume orientation
        ress_np = np.transpose(ress_sf.numpy(), inv_perm)

        # Perceptual metrics
        if args.use_gpu_perceptual:
            pred_t = torch.from_numpy(ress_np)[None, None].float().cuda()
            gt_t = torch.from_numpy(im2)[None, None].float().cuda()
        else:
            pred_t = torch.from_numpy(ress_np)[None, None].float()
            gt_t = torch.from_numpy(im2)[None, None].float()

        df["perc_10"].append(model10(pred_t, gt_t).item())
        df["perc_50"].append(model50(pred_t, gt_t).item())

        # Move back to numpy for PSNR/SSIM
        df["psnr"].append(peak_signal_noise_ratio(im2, ress_np, data_range=1))
        df["ssim"].append(structural_similarity(im2, ress_np, data_range=1))
        df["patient"].append(i)

        # Brain-only metrics (gt>0 mask)
        f_true = im2.flatten()
        f_gen = ress_np.flatten()
        f_gen = f_gen[f_true > 0]
        print("brain voxel ratio:", len(f_gen) / len(f_true))
        f_true = f_true[f_true > 0]

        df["psnr_brain"].append(peak_signal_noise_ratio(f_true, f_gen, data_range=1))
        df["ssim_brain"].append(structural_similarity(f_true, f_gen, data_range=1))

        # "No corrupt" region metrics
        # Prefer axis-specific bottom border; fallback to old bot_first_slice if missing.
        if f"bot_{args.axis}" in info.columns:
            shifting = int(info.loc[info["7T_idx"] == key, f"bot_{args.axis}"].iloc[0])
        elif "bot_first_slice" in info.columns:
            shifting = int(info.loc[info["7T_idx"] == key, "bot_first_slice"].iloc[0])
        else:
            shifting = 0

        ress_s = slice_along_axis(ress_np, args.axis, shifting)
        im2_s = slice_along_axis(im2, args.axis, shifting)

        f_true2 = im2_s.flatten()
        f_gen2 = ress_s.flatten()
        f_gen2 = f_gen2[f_true2 > 0]
        f_true2 = f_true2[f_true2 > 0]

        df["psnr_no_corrupt"].append(peak_signal_noise_ratio(f_true2, f_gen2, data_range=1))
        df["ssim_no_corrupt"].append(structural_similarity(f_true2, f_gen2, data_range=1))

        # Save generated volume
        #out_path = generated_7T / f"{i}_t1_generated.nii.gz"
        out_path = generated_7T / f"{i}_t1_generated__{tag}.nii.gz"
        
        nib.save(nib.Nifti1Image(ress_np, af, header=hd), str(out_path))
        print(f"Saved: {out_path}")

    # Append mean row
    if len(df["patient"]) > 0:
        df["psnr"].append(float(np.mean(np.array(df["psnr"]))))
        df["ssim"].append(float(np.mean(np.array(df["ssim"]))))

        df["psnr_brain"].append(float(np.mean(np.array(df["psnr_brain"]))))
        df["ssim_brain"].append(float(np.mean(np.array(df["ssim_brain"]))))

        df["perc_10"].append(float(np.mean(np.array(df["perc_10"]))))
        df["perc_50"].append(float(np.mean(np.array(df["perc_50"]))))

        df["psnr_no_corrupt"].append(float(np.mean(np.array(df["psnr_no_corrupt"]))))
        df["ssim_no_corrupt"].append(float(np.mean(np.array(df["ssim_no_corrupt"]))))

        df["patient"].append(0)

    df_pd = pd.DataFrame(data=df)
    df_pd.to_csv(metrics_csv_path, index=False)
    print(f"Metrics saved to: {metrics_csv_path}")


if __name__ == "__main__":
    main()
