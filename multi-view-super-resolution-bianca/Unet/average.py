#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Batch average of 3-axis NIfTI predictions.")
    p.add_argument("--dir-axis0", required=True, help="Directory for axis0 outputs (e.g., coronal).")
    p.add_argument("--dir-axis1", required=True, help="Directory for axis1 outputs (e.g., axial).")
    p.add_argument("--dir-axis2", required=True, help="Directory for axis2 outputs (e.g., sagittal).")
    p.add_argument("--out-dir", required=True, help="Output directory for averaged volumes.")
    p.add_argument("--strict-affine", action="store_true", help="Fail if affine mismatch (recommended).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return p.parse_args()


def build_id_map(folder: Path):
    """
    Build a map: subject_id (e.g., '126') -> file_path
    Expected filename example:
      126_t1_generated__axis0_on_UnetModel0.nii.gz
    We extract the leading integer before '_t1_generated__'.
    """
    id_map = {}
    pattern = re.compile(r"^(\d+)_t1_generated__.*\.nii(\.gz)?$")
    for f in folder.glob("*.nii*"):
        m = pattern.match(f.name)
        if not m:
            continue
        sid = m.group(1)
        id_map[sid] = f
    return id_map


def main():
    args = parse_args()

    d0 = Path(args.dir_axis0)
    d1 = Path(args.dir_axis1)
    d2 = Path(args.dir_axis2)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    map0 = build_id_map(d0)
    map1 = build_id_map(d1)
    map2 = build_id_map(d2)

    common_ids = sorted(set(map0.keys()) & set(map1.keys()) & set(map2.keys()))
    missing0 = sorted(set(map1.keys()) & set(map2.keys()) - set(map0.keys()))
    missing1 = sorted(set(map0.keys()) & set(map2.keys()) - set(map1.keys()))
    missing2 = sorted(set(map0.keys()) & set(map1.keys()) - set(map2.keys()))

    print(f"[INFO] axis0 files: {len(map0)}  axis1 files: {len(map1)}  axis2 files: {len(map2)}")
    print(f"[INFO] common subjects to average: {len(common_ids)}")
    if missing0:
        print(f"[WARN] missing in axis0 (show up to 10): {missing0[:10]}")
    if missing1:
        print(f"[WARN] missing in axis1 (show up to 10): {missing1[:10]}")
    if missing2:
        print(f"[WARN] missing in axis2 (show up to 10): {missing2[:10]}")

    n_done = 0
    for sid in common_ids:
        f0 = map0[sid]
        f1 = map1[sid]
        f2 = map2[sid]

        out_path = out_dir / f"{sid}_t1_generated__average.nii.gz"
        if out_path.exists() and (not args.overwrite):
            print(f"[SKIP] exists: {out_path.name}")
            continue

        # Load volumes
        img0 = nib.load(str(f0))
        img1 = nib.load(str(f1))
        img2 = nib.load(str(f2))

        data0 = img0.get_fdata()
        data1 = img1.get_fdata()
        data2 = img2.get_fdata()

        # Check shapes
        if not (data0.shape == data1.shape == data2.shape):
            raise ValueError(
                f"Shape mismatch for subject {sid}: "
                f"{data0.shape}, {data1.shape}, {data2.shape}"
            )

        # Affine check (recommended)
        if args.strict_affine:
            if not (np.allclose(img0.affine, img1.affine) and np.allclose(img0.affine, img2.affine)):
                raise ValueError(f"Affine mismatch for subject {sid}: inputs may not be aligned")

        # Voxel-wise average
        avg = (data0 + data1 + data2) / 3.0

        # Save using axis0's affine/header (any one is fine if they match)
        out_img = nib.Nifti1Image(avg, img0.affine, img0.header)
        nib.save(out_img, str(out_path))
        n_done += 1

        if n_done % 20 == 0:
            print(f"[INFO] done {n_done}/{len(common_ids)} ...")

    print(f"[INFO] finished. saved: {n_done} files to {out_dir}")


if __name__ == "__main__":
    main()
