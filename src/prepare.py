"""
Task 1 – Load and prepare the data.

Steps:
  1.1  Load raw signals from data/ and print shapes.
  1.2  Convert labels from 1-6 to 0-5; report windows per activity.
  1.3  Plot one window per activity (calls plot_windows.py).
  1.4  Validation split by person: VAL_SUBJECTS = [3, 15, 25, 30].
  1.5  Standardise each of the 9 channels (mean/std from training windows only).
  1.6  Convert to torch tensors.
"""

import os
import sys
import numpy as np
import torch

# Allow running from repo root: python src/prepare.py
sys.path.insert(0, os.path.dirname(__file__))

from load_data import load_raw, CHANNELS

# ------------------------------------------------------------------
# 1.1  Load raw data
# ------------------------------------------------------------------
def load_and_prepare(data_dir="data"):
    raw = load_raw(data_dir)

    # Print shapes
    for split in ["train", "test"]:
        d = raw[split]
        print(f"{split}: X {d['X'].shape}, y {d['y'].shape}, "
              f"subjects {sorted(set(d['subject'].tolist()))}")
    print("activities:", raw["activity_names"])

    # ------------------------------------------------------------------
    # 1.2  Convert labels 1-6  →  0-5
    # ------------------------------------------------------------------
    for split in ["train", "test"]:
        raw[split]["y"] = raw[split]["y"] - 1  # now 0..5

    train_all = raw["train"]
    test      = raw["test"]
    names     = raw["activity_names"]  # list of 6 strings

    # Report windows per activity (original train_all)
    print("\nWindows per activity in original train split:")
    for k, name in enumerate(names):
        n = int(np.sum(train_all["y"] == k))
        print(f"  {k+1} {name}: {n}")

    print("\nWindows per activity in test split:")
    for k, name in enumerate(names):
        n = int(np.sum(test["y"] == k))
        print(f"  {k+1} {name}: {n}")

    # ------------------------------------------------------------------
    # 1.4  Validation split by person
    # ------------------------------------------------------------------
    VAL_SUBJECTS = [3, 15, 25, 30]
    is_val = np.isin(train_all["subject"], VAL_SUBJECTS)

    val = {
        "X":       train_all["X"][is_val],
        "y":       train_all["y"][is_val],
        "subject": train_all["subject"][is_val],
    }
    train = {
        "X":       train_all["X"][~is_val],
        "y":       train_all["y"][~is_val],
        "subject": train_all["subject"][~is_val],
    }

    print(f"\nAfter person-based validation split:")
    print(f"  Train   : {train['X'].shape[0]} windows, "
          f"subjects {sorted(set(train['subject'].tolist()))}")
    print(f"  Val     : {val['X'].shape[0]} windows, "
          f"subjects {sorted(set(val['subject'].tolist()))}")
    print(f"  Test    : {test['X'].shape[0]} windows, "
          f"subjects {sorted(set(test['subject'].tolist()))}")

    # ------------------------------------------------------------------
    # 1.5  Standardise each channel (stats from training windows only)
    # ------------------------------------------------------------------
    # train['X'] shape: (N_train, 128, 9)
    # Compute mean/std over all time steps of training windows: shape (9,)
    X_tr_flat = train["X"].reshape(-1, 9)   # (N_train * 128, 9)
    ch_mean   = X_tr_flat.mean(axis=0)      # (9,)
    ch_std    = X_tr_flat.std(axis=0)       # (9,)

    print("\nChannel means  (training):", ch_mean)
    print("Channel stds   (training):", ch_std)

    # Apply standardisation
    for split_dict in [train, val, test]:
        split_dict["X"] = (split_dict["X"] - ch_mean) / ch_std

    # ------------------------------------------------------------------
    # 1.6  Convert to torch tensors
    # ------------------------------------------------------------------
    X_train = torch.tensor(train["X"], dtype=torch.float32)
    y_train = torch.tensor(train["y"], dtype=torch.long)
    X_val   = torch.tensor(val["X"],   dtype=torch.float32)
    y_val   = torch.tensor(val["y"],   dtype=torch.long)
    X_test  = torch.tensor(test["X"],  dtype=torch.float32)
    y_test  = torch.tensor(test["y"],  dtype=torch.long)

    print(f"\nTensor shapes:")
    print(f"  X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"  X_val   {X_val.shape},   y_val   {y_val.shape}")
    print(f"  X_test  {X_test.shape},  y_test  {y_test.shape}")

    return (X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            names, ch_mean, ch_std)


if __name__ == "__main__":
    # Run plot_windows first (Task 1.3)
    import subprocess
    os.makedirs("results", exist_ok=True)
    subprocess.run([sys.executable, "src/plot_windows.py"], check=False)

    load_and_prepare()
