"""
Task 3 – Train HARNet.

Settings (fixed):
  Loss           : cross-entropy
  Optimiser      : Adam, lr = 1e-3
  Batch size     : 64
  Epochs         : 30
  Gradient clip  : total norm 1.0
  Shuffle        : every epoch

After every epoch: record train loss, val loss, val accuracy.
Save parameters from epoch with lowest validation loss.
"""

import os, sys, time, platform
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from prepare import load_and_prepare
from model   import HARNet


def train(data_dir="data", results_dir="results", epochs=30,
          batch_size=64, lr=1e-3, clip_norm=1.0, seed=0):

    os.makedirs(results_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     names, ch_mean, ch_std) = load_and_prepare(data_dir)

    # ── Model ─────────────────────────────────────────────────
    torch.manual_seed(seed)
    model = HARNet()

    # Verify parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {total_params:,}")
    assert total_params == 46_982, f"Expected 46,982 got {total_params}"

    # ── Initial loss (before training) ────────────────────────
    model.eval()
    with torch.no_grad():
        init_loss_total = 0.0
        n_init_batches = 0
        for start in range(0, X_train.shape[0], batch_size):
            xb = X_train[start : start + batch_size]
            yb = y_train[start : start + batch_size]
            logits0, _ = model(xb)
            init_loss_total += F.cross_entropy(logits0, yb).item()
            n_init_batches += 1
        init_loss = init_loss_total / n_init_batches
    print(f"Initial loss (before training): {init_loss:.4f}  (reference ln6 ≈ 1.7918)")

    # ── Optimiser ─────────────────────────────────────────────
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # ── Training loop ─────────────────────────────────────────
    N_train = X_train.shape[0]
    hardware = f"CPU ({platform.processor() or platform.machine()})"
    history = {
        "train_loss": [], "val_loss": [], "val_acc": [],
        "init_loss": init_loss, "total_params": total_params,
        "hardware": hardware,
    }

    best_val_loss   = float("inf")
    best_epoch      = -1
    best_state      = None
    t0              = time.time()

    for epoch in range(1, epochs + 1):
        # Shuffle training data each epoch
        perm = torch.randperm(N_train)
        X_tr_shuf = X_train[perm]
        y_tr_shuf = y_train[perm]

        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, N_train, batch_size):
            xb = X_tr_shuf[start : start + batch_size]
            yb = y_tr_shuf[start : start + batch_size]

            optimiser.zero_grad()
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()

            # 3.1  Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            optimiser.step()
            epoch_loss += loss.item()
            n_batches  += 1

        train_loss = epoch_loss / n_batches

        # ── Validation ────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_loss_total = 0.0
            val_correct    = 0
            n_val_batches  = 0

            for start in range(0, X_val.shape[0], batch_size):
                xb = X_val[start : start + batch_size]
                yb = y_val[start : start + batch_size]
                logits, _ = model(xb)
                val_loss_total += F.cross_entropy(logits, yb).item()
                val_correct    += (logits.argmax(dim=1) == yb).sum().item()
                n_val_batches  += 1

            val_loss = val_loss_total / n_val_batches
            val_acc  = val_correct / X_val.shape[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 3.3  Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | "
              f"val_acc {val_acc*100:.2f}%")

    elapsed = time.time() - t0
    history["best_epoch"] = best_epoch
    history["best_train_loss"] = history["train_loss"][best_epoch - 1]
    history["best_val_loss"] = best_val_loss
    history["best_val_acc"] = history["val_acc"][best_epoch - 1]
    history["elapsed_s"] = elapsed
    print(f"\nTraining finished in {elapsed:.1f}s ({hardware})")
    print(f"Best epoch: {best_epoch}  "
          f"(train_loss {history['best_train_loss']:.4f}, "
          f"val_loss {best_val_loss:.4f}, "
          f"val_acc {history['best_val_acc']*100:.2f}%)")

    # Save best checkpoint
    model_path = os.path.join(results_dir, "best_model.pt")
    torch.save(best_state, model_path)
    print(f"Saved best model to {model_path}")

    # ── 3.2  Plot loss and accuracy curves ────────────────────
    epochs_range = range(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_range, history["train_loss"], label="Train loss")
    ax1.plot(epochs_range, history["val_loss"],   label="Val loss")
    ax1.axvline(best_epoch, color="red", linestyle="--", label=f"Best epoch {best_epoch}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()

    ax2.plot(epochs_range, [a * 100 for a in history["val_acc"]], color="green")
    ax2.axvline(best_epoch, color="red", linestyle="--", label=f"Best epoch {best_epoch}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation accuracy (%)")
    ax2.set_title("Validation Accuracy")
    ax2.legend()

    fig.tight_layout()
    curve_path = os.path.join(results_dir, "training_curves.png")
    fig.savefig(curve_path, dpi=130)
    print(f"Saved training curves to {curve_path}")

    # Save history and stats
    np.save(os.path.join(results_dir, "training_history.npy"), history)

    # Save channel stats for the front-end / test
    np.save(os.path.join(results_dir, "ch_mean.npy"), ch_mean)
    np.save(os.path.join(results_dir, "ch_std.npy"),  ch_std)

    return model, best_state, history, best_epoch, (X_test, y_test, names)


if __name__ == "__main__":
    train()
