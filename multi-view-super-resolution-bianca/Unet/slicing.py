#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strict parameterised rewrite of the original ano_3dim.py

Goals:
- Preserve the ORIGINAL slicing / cropping / normalization logic as closely as possible
- Keep argparse
- Keep batch saving
- Keep support for axis 0/1/2
- Keep original CSV column names and condition semantics

Compared with the original script:
- Supports CLI args
- Supports saving every N subjects
- Supports axis 0/1/2 while reproducing the original logic for each axis
- Saves config.json for reproducibility

Original logic preserved:
1. Subject order generation: permutation over a contiguous candidate range, then +offset
2. Use fixed global border_bot / border_top for the final cropped training block
3. Use per-subject bot_k / top_k ROI only for clipping / normalization statistics
4. 3T input is cropped with extra neighbor margin along slice axis
5. 7T target is cropped without that extra margin
6. Sliding windows are built exactly like the original:
   - axis 0: x[j] = Img_n_3T[j:j+n_colors], y[j] = Img_n_7T[j]
   - axis 1: x[j] = transpose(Img_n_3T[:, j:j+n_colors], [1,0,2]), y[j] = transpose(Img_n_7T, [1,0,2])[j]
   - axis 2: x[j] = transpose(Img_n_3T[:, :, j:j+n_colors], [2,0,1]), y[j] = transpose(Img_n_7T, [2,0,1])[j]
7. Context "position" follows the original formula style:
   2*(np.arange(full_dim)-mid)[bot:top+1] / n_slices
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

def percentile_clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    lo_v, hi_v = np.percentile(x, [lo, hi])
    return np.clip(x, lo_v, hi_v)


def norm_min_max(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    xmin = np.min(ref)
    xmax = np.max(ref)
    if xmax <= xmin:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def norm_std_mean(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mu = float(np.mean(ref))
    sd = float(np.std(ref))
    if sd == 0:
        return np.zeros_like(x)
    return (x - mu) / sd


def save_config(cfg: dict, out_dir: Path):
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def get_default_borders():
    # Original borders from the first script#################################################
    border_bot = [50, 61, 26]
    border_top = [337, 302, 247]
    return border_bot, border_top


def compute_axis_crop_params(axis: int, n_colors: int, border_bot, border_top):
    """
    Reproduce the original cropping logic for arbitrary axis.

    Original for axis=1:
        3T: [b0:t0+1, bot-nc//2 : top+1+nc//2, b2:t2+1]
        7T: [b0:t0+1, bot       : top+1,       b2:t2+1]

    Generalized:
    - non-slice axes use fixed border_bot[k] : border_top[k]+1
    - slice axis:
        3T uses [bot - n_colors//2 : top+1 + n_colors//2]
        7T uses [bot : top+1]

    Also returns:
    - bot_axis = border_bot[axis] - 1
    - top_axis = border_top[axis] + 1
      exactly like the original style
    """
    bot_axis = border_bot[axis] - 1
    top_axis = border_top[axis] + 1

    crop3 = []
    crop7 = []
    for k in range(3):
        if k == axis:
            crop3.append(slice(bot_axis - n_colors // 2, top_axis + 1 + n_colors // 2))
            crop7.append(slice(bot_axis, top_axis + 1))
        else:
            crop3.append(slice(border_bot[k], border_top[k] + 1))
            crop7.append(slice(border_bot[k], border_top[k] + 1))

    return tuple(crop3), tuple(crop7), bot_axis, top_axis


def build_position_context(axis_len_full: int, bot_axis: int, top_axis: int, mid: float, n_slices: int):
    """
    Match original:
        2*(np.arange(0, full_dim)-mid)[bot:top+1] / n_slices
    """
    arr = 2 * (np.arange(0, axis_len_full) - mid) / n_slices
    return arr[bot_axis:top_axis + 1]


def pick_subjects_strict(info: pd.DataFrame,
                         usage: str,
                         skip_list,
                         perm_range: int,
                         idx_offset: int,
                         train_size: int,
                         seed: int):
    """
    Original subject selection style:
        ano_idx = np.random.permutation(179)
        idx7t = ano_idx[j] + 126

    Here generalized by args.
    """
    np.random.seed(seed)
    random.seed(seed)

    ano_idx = np.random.permutation(perm_range)

    subjects = []
    for j in range(len(ano_idx)):
        idx7t = int(ano_idx[j]) + idx_offset
        row = info.loc[info["7T_idx"] == f"7T_015_BioFINDER_{idx7t}"]
        if len(row) == 0:
            continue
        if idx7t in skip_list:
            continue
        if usage != "all":
            if row["usage"].iloc[0] != usage:
                continue
        subjects.append(idx7t)
        if len(subjects) >= train_size:
            break
    return subjects


# -----------------------------
# Argument parsing
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Strict parameterised rewrite of original ano_3dim.py with batch saving."
    )

    # Paths
    p.add_argument("--big-path", type=Path, required=True,
                   help="Base path containing patient CSV etc.")
    p.add_argument("--parent-3t", type=Path, required=True,
                   help="Parent dir of 3T folders")
    p.add_argument("--parent-7t", type=Path, required=True,
                   help="Parent dir of 7T nifti files")
    p.add_argument("--save-path", type=Path, required=True,
                   help="Where outputs will be saved")
    p.add_argument("--patient-csv", type=str, default="patient_info.csv")

    # Data logic
    p.add_argument("--axis", type=int, default=1, choices=[0, 1, 2],
                   help="Slice axis (0/1/2)")
    p.add_argument("--n-in-slices", type=int, required=True,
                   help="Number of input slices (channels), same meaning as original sys.argv[1]")
    p.add_argument("--conditions", nargs="*", default=["Sex", "Age", "Diagnostic", "position"],
                   help="Kept for record/config; actual values follow original semantics")
    p.add_argument("--normalization-method", type=str, default="min_max",
                   choices=["min_max", "std_mean"])
    p.add_argument("--clip-low", type=float, default=1.0)
    p.add_argument("--clip-high", type=float, default=99.0)
    p.add_argument("--disable-clip", action="store_true",
                   help="If set, do not percentile-clip before estimating normalization stats")

    # Subject selection
    p.add_argument("--usage", type=str, default="train", choices=["train", "val", "test", "all"])
    p.add_argument("--train-size", type=int, default=138)
    p.add_argument("--seed", type=int, default=2025)

    # Original index scheme
    p.add_argument("--perm-range", type=int, default=179,
                   help="Original was 179")
    p.add_argument("--idx-offset", type=int, default=126,
                   help="Original added +126 after permutation")
    p.add_argument("--skip", type=int, nargs="*",
                   default=[203, 204, 223, 248, 274, 288, 300, 184, 192])

    # Border control
    p.add_argument("--border-bot", type=int, nargs=3, default=[50, 61, 26])
    p.add_argument("--border-top", type=int, nargs=3, default=[337, 302, 247])

    # Save / dtype
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument("--run-tag", type=str, default="")
    p.add_argument("--batch-subjects", type=int, default=10,
                   help="Save every N subjects")

    # Naming compatibility
    p.add_argument("--save-index", type=int, default=1,
                   help="Equivalent to original n_save")

    return p.parse_args()


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    t0 = time.time()

    save_dir = args.save_path
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "images").mkdir(parents=True, exist_ok=True)

    # Read CSV
    csv_path = args.big_path / args.patient_csv
    info = pd.read_csv(csv_path)

    skip_set = set(args.skip)
    n_colors = int(args.n_in_slices)
    axis = int(args.axis)
    border_bot = list(args.border_bot)
    border_top = list(args.border_top)

    # Determine cropped training block
    crop3, crop7, bot_axis, top_axis = compute_axis_crop_params(
        axis=axis,
        n_colors=n_colors,
        border_bot=border_bot,
        border_top=border_top
    )
    n_slices = top_axis - bot_axis + 1  # exactly like original effective count

    logging.info(f"axis={axis}, n_in_slices={n_colors}")
    logging.info(f"border_bot={border_bot}, border_top={border_top}")
    logging.info(f"effective n_slices={n_slices}")
    logging.info(f"3T crop={crop3}")
    logging.info(f"7T crop={crop7}")

    # Subject selection strictly following original style
    subjects = pick_subjects_strict(
        info=info,
        usage=args.usage,
        skip_list=skip_set,
        perm_range=args.perm_range,
        idx_offset=args.idx_offset,
        train_size=args.train_size,
        seed=args.seed
    )

    if not subjects:
        raise RuntimeError("No subjects selected. Check CSV, usage, perm-range, idx-offset, skip list.")

    logging.info(f"Selected subjects: {len(subjects)}")

    # dtype
    dtype = np.float16 if args.dtype == "float16" else np.float32

    # Accumulators for batch saving
    xs = []
    ys = []
    ctx_rows = []
    range_rows = []   # keep per-subject, same spirit as original
    infos_rows = []   # per-subject bot/top records
    batch_idx = 0
    total_samples = 0
    total_subjects_saved = 0
    first_input_shape = None
    first_output_shape = None

    def save_batch():
        nonlocal xs, ys, ctx_rows, range_rows, infos_rows
        nonlocal batch_idx, total_samples, total_subjects_saved
        nonlocal first_input_shape, first_output_shape

        if len(xs) == 0:
            return

        batch_idx += 1

        X = np.stack(xs, axis=0).astype(dtype, copy=False)          # (N, C, H, W)
        Y = np.stack(ys, axis=0).astype(dtype, copy=False)          # (N, 1, H, W)
        C = np.asarray(ctx_rows, dtype=dtype).reshape(len(ctx_rows), 4, 1)
        R = np.asarray(range_rows, dtype=dtype)

        if first_input_shape is None:
            first_input_shape = list(X.shape[1:])
        if first_output_shape is None:
            first_output_shape = list(Y.shape[1:])

        # Filename style: keep second-script flexibility, but easy to read
        tag_base = f"_a{axis}_s{n_colors}"
        if args.run_tag:
            tag_base = f"_{args.run_tag}{tag_base}"
        batch_tag = f"{tag_base}_b{batch_idx}"

        np.save(save_dir / f"inputs{batch_tag}.npy", X, allow_pickle=False)
        np.save(save_dir / f"outputs{batch_tag}.npy", Y, allow_pickle=False)
        np.save(save_dir / f"contexts{batch_tag}.npy", C, allow_pickle=False)
        np.save(save_dir / f"ranges{batch_tag}.npy", R, allow_pickle=False)

        infos_df = pd.DataFrame(infos_rows)
        infos_df.to_csv(save_dir / f"infos{batch_tag}.csv", index=False)

        total_samples += X.shape[0]
        total_subjects_saved += len(range_rows)

        print(f"[INFO] Saved batch {batch_idx}: "
              f"{len(range_rows)} subjects, {X.shape[0]} samples. "
              f"Total so far: {total_subjects_saved} subjects, {total_samples} samples")

        xs.clear()
        ys.clear()
        ctx_rows.clear()
        range_rows.clear()
        infos_rows.clear()

    # Iterate subjects
    for si, idx7t in enumerate(subjects):
        tic = time.time()

        row_df = info.loc[info["7T_idx"] == f"7T_015_BioFINDER_{idx7t}"]
        if len(row_df) != 1:
            logging.warning(f"Skip {idx7t}: expected 1 row, got {len(row_df)}")
            continue
        row = row_df.iloc[0]

        # Original per-subject ROI borders
        bot0 = int(row["bot_0"])
        bot1 = int(row["bot_1"])
        bot2 = int(row["bot_2"])
        top0 = int(row["top_0"])
        top1 = int(row["top_1"])
        top2 = int(row["top_2"])

        # For info CSV, original used bot_abs based on axis=1 only.
        # Generalize to the selected axis.
        bot_axis_subject = int(row[f"bot_{axis}"])
        top_axis_subject = int(row[f"top_{axis}"])
        bot_first_slice = int(row["bot_first_slice"]) if "bot_first_slice" in row.index else bot_axis_subject
        bot_abs = max(bot_axis_subject, bot_first_slice)

        # Build 3T id exactly like original
        date = str(row["closest_3T"])
        dates = date.split("-")
        if len(dates) != 3:
            logging.warning(f"Skip {idx7t}: malformed closest_3T={date}")
            continue

        mid_3t = str(row["mid"])
        id_3t = mid_3t + "__" + dates[0] + dates[1] + dates[2]

        p3t = args.parent_3t / id_3t / "brain_corrected_registered.nii.gz"
        p7t = args.parent_7t / f"{idx7t}_t1_brain3.nii.gz"

        try:
            img3t_nii = nib.load(str(p3t))
            img3t = img3t_nii.get_fdata()
        except Exception as e:
            logging.warning(f"Skip {idx7t}: failed to read 3T {p3t}: {e}")
            continue

        try:
            img7t = nib.load(str(p7t)).get_fdata()
        except Exception as e:
            logging.warning(f"Skip {idx7t}: failed to read 7T {p7t}: {e}")
            continue

        # Save placeholder header/affine, same spirit as original
        nib.save(
            nib.Nifti1Image(
                np.zeros((1, 1, 1), dtype=np.float32),
                header=img3t_nii.header,
                affine=img3t_nii.affine
            ),
            str(save_dir / "images" / f"hd_af_{idx7t}.nii.gz")
        )

        # Per-subject ROI for normalization statistics, exactly from bot_k/top_k
        roi_slices = (
            slice(bot0, top0 + 1),
            slice(bot1, top1 + 1),
            slice(bot2, top2 + 1),
        )
        img3t_small = img3t[roi_slices]
        img7t_small = img7t[roi_slices]

        # Final cropped training blocks, exactly from fixed borders + neighbor margin on slice axis
        try:
            img3t_crop = img3t[crop3]
            img7t_crop = img7t[crop7]
        except Exception as e:
            logging.warning(f"Skip {idx7t}: crop failed: {e}")
            continue

        # Safety checks: shape along axis should match original intended logic
        if img7t_crop.shape[axis] != n_slices:
            logging.warning(
                f"Skip {idx7t}: img7t_crop.shape[{axis}]={img7t_crop.shape[axis]} "
                f"!= expected n_slices={n_slices}"
            )
            continue


        expected_3t_len = n_slices + 2 * (n_colors // 2)
        if img3t_crop.shape[axis] != expected_3t_len:
            logging.warning(
                f"Skip {idx7t}: img3t_crop.shape[{axis}]={img3t_crop.shape[axis]} "
                f"!= expected {expected_3t_len}. "
                f"This may happen if borders exceed image bounds."
            )
            continue


        # Clipping stats from small ROI
        if not args.disable_clip:
            v3t = percentile_clip(img3t_small, args.clip_low, args.clip_high)
            v7t = percentile_clip(img7t_small, args.clip_low, args.clip_high)
        else:
            v3t = img3t_small.copy()
            v7t = img7t_small.copy()

        # Normalize cropped training blocks using stats from ROI, same as original
        if args.normalization_method == "min_max":
            img3t_n = norm_min_max(img3t_crop, v3t).astype(dtype)
            img7t_n = norm_min_max(img7t_crop, v7t).astype(dtype)
        else:
            img3t_n = norm_std_mean(img3t_crop, v3t).astype(dtype)
            img7t_n = norm_std_mean(img7t_crop, v7t).astype(dtype)

        # Range per subject, same spirit as original
        subj_range = float(np.max(img7t_n) - np.min(img7t_n))
        range_rows.append(subj_range)

        # Original condition semantics
        # Note: original array order was effectively [Age, Sex, Diagnostic, position]
        # even though settings["conditions"] listed ["Sex","Age","Diagnostic","position"].
        # We preserve the ORIGINAL ASSIGNMENT, not the mislabeled names.
        age_val = float(row["age"])
        sex_val = float(row["gender_baseline_variable"])
        diag_val = float(row["cognitive_status_baseline_variable"] in ["SCD", "Normal"])

        mid = (bot_axis_subject + top_axis_subject) / 2.0
        pos_ctx = build_position_context(
            axis_len_full=img3t.shape[axis],
            bot_axis=bot_axis,
            top_axis=top_axis,
            mid=mid,
            n_slices=n_slices
        )

        if len(pos_ctx) != n_slices:
            logging.warning(
                f"Skip {idx7t}: position context length {len(pos_ctx)} != n_slices {n_slices}"
            )
            range_rows.pop()
            continue

        # Save per-subject infos
        infos_rows.append({
            f"bot_{axis}": bot_axis_subject,
            f"top_{axis}": top_axis_subject,
            "bot_abs": bot_abs,
            "idx7t": idx7t
        })

        # Build samples exactly like original
        if axis == 0:
            # img3t_n shape: [n_slices+n_colors, H, W]
            # img7t_n shape: [n_slices, H, W]
            for j in range(n_slices):
                x_j = img3t_n[j:j + n_colors]              # (C,H,W)
                y_j = img7t_n[j][None, ...]                # (1,H,W)

                xs.append(x_j)
                ys.append(y_j)
                ctx_rows.append([age_val, sex_val, diag_val, float(pos_ctx[j])])

        elif axis == 1:
            # img3t_n shape: [H, n_slices+n_colors, W]
            # x_j = transpose(img3t_n[:, j:j+n_colors], [1,0,2])
            # y   = transpose(img7t_n, [1,0,2])
            y_all = np.transpose(img7t_n, axes=[1, 0, 2])   # (n_slices, H, W)
            for j in range(n_slices):
                x_j = np.transpose(img3t_n[:, j:j + n_colors], axes=[1, 0, 2])   # (C,H,W)
                y_j = y_all[j][None, ...]                                          # (1,H,W)

                xs.append(x_j)
                ys.append(y_j)
                ctx_rows.append([age_val, sex_val, diag_val, float(pos_ctx[j])])

        else:  # axis == 2
            # img3t_n shape: [H, W, n_slices+n_colors]
            # x_j = transpose(img3t_n[:, :, j:j+n_colors], [2,0,1])
            # y   = transpose(img7t_n, [2,0,1])
            y_all = np.transpose(img7t_n, axes=[2, 0, 1])   # (n_slices, H, W)
            for j in range(n_slices):
                x_j = np.transpose(img3t_n[:, :, j:j + n_colors], axes=[2, 0, 1])   # (C,H,W)
                y_j = y_all[j][None, ...]                                             # (1,H,W)

                xs.append(x_j)
                ys.append(y_j)
                ctx_rows.append([age_val, sex_val, diag_val, float(pos_ctx[j])])

        toc = time.time()
        print(f"[INFO] subject {idx7t} done in {toc - tic:.2f}s")

        # Save every N subjects
        if ((si + 1) % args.batch_subjects == 0) or ((si + 1) == len(subjects)):
            save_batch()

    # Save config summary
    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    save_config({
        "args": args_dict,
        "original_logic_preserved": True,
        "condition_assignment_order": ["age", "gender_baseline_variable", "cognitive_status_baseline_variable_binary", "position"],
        "condition_names_argument": args.conditions,
        "effective_n_slices": int(n_slices),
        "input_shape": first_input_shape,
        "output_shape": first_output_shape,
        "batches": int(batch_idx),
        "total_samples": int(total_samples),
        "total_subjects_saved": int(total_subjects_saved),
    }, save_dir)

    print(f"[Done] total subjects saved: {total_subjects_saved}, total samples: {total_samples}, batches: {batch_idx}")
    print(f"[Done] elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
