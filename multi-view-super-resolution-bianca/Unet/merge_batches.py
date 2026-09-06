#!/usr/bin/env python3
"""
Merge batched outputs from ano_3dim_param.py
into single combined .npy files.
"""

import numpy as np
from pathlib import Path
import argparse
import re


def batch_key(path: Path):
    m = re.search(r"_b(\d+)\.npy$", path.name)
    if not m:
        raise ValueError(f"Cannot parse batch index from filename: {path.name}")
    return int(m.group(1))


def concat_and_save(files, out_path):
    if not files:
        print(f"[WARN] No files found for {out_path.name}")
        return
    print(f"[INFO] Merging {len(files)} parts into {out_path.name}")
    for f in files:
        print(f"   {f.name}")
    arrays = [np.load(f) for f in files]
    merged = np.concatenate(arrays, axis=0)
    np.save(out_path, merged)
    print(f"[INFO] Saved {out_path.name}, shape={merged.shape}, size={merged.nbytes/1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Merge batched .npy files from ano_3dim_param.py")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing batched .npy files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder to save merged .npy files")
    args = parser.parse_args()

    in_dir = args.input_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = {
        "inputs": "inputs*_b*.npy",
        "outputs": "outputs*_b*.npy",
        "contexts": "contexts*_b*.npy",
        "ranges": "ranges*_b*.npy",
    }

    for key, pat in patterns.items():
        files = sorted(in_dir.glob(pat), key=batch_key)
        concat_and_save(files, out_dir / f"{key}1_s3.npy")
        print(f"[DONE] {key} merged and saved to {out_dir / f'{key}1_s3.npy'}")

    print("\n✅ All merged successfully.")


if __name__ == "__main__":
    main()

