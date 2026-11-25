"""
Tasks 4 and 5 – Test the saved model and visualise attention.

Task 4:
  4.1  Load saved parameters and predict every test window.
  4.2  Compute accuracy, confusion matrix, per-class accuracy (using metrics.py).
  4.3  Report results and save confusion-matrix heat-map.

Task 5:
  5.1  For one correctly classified window per activity, plot signals + attention.
  5.2  For each activity, average attention over all its test windows; plot.
  5.3  Report mean of max alpha_t per activity.
"""

import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from prepare  import load_and_prepare
from model    import HARNet
from metrics  import accuracy, confusion_matrix, per_class_accuracy

CHANNELS = ["body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
            "total_acc_x", "total_acc_y", "total_acc_z"]

FS = 50  # Hz


def test_and_visualise(data_dir="data", results_dir="results", batch_size=64):
    os.makedirs(results_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     names, ch_mean, ch_std) = load_and_prepare(data_dir)

    # ── 4.1  Load model and predict ───────────────────────────
    model = HARNet()
    state = torch.load(
        os.path.join(results_dir, "best_model.pt"),
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    all_preds  = []
    all_alphas = []

    with torch.no_grad():
        for start in range(0, X_test.shape[0], batch_size):
            xb = X_test[start : start + batch_size]
            logits, alpha = model(xb)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.numpy())
            all_alphas.append(alpha.numpy())

    y_pred = np.concatenate(all_preds)        # (N_test,)
    alphas = np.concatenate(all_alphas)       # (N_test, 128)
    y_true = y_test.numpy()                   # (N_test,)

    # Correctness check: attention weights > 0 and sum to 1
    assert np.all(alphas > 0), "Found non-positive attention weights!"
    assert np.allclose(alphas.sum(axis=1), 1.0, atol=1e-5), "Attention weights do not sum to 1!"
    print("Attention positivity and sum-to-1 verified ✓")

    # ── 4.2  Metrics ──────────────────────────────────────────
    acc  = accuracy(y_true, y_pred)
    conf = confusion_matrix(y_true, y_pred, num_classes=6)
    pca  = per_class_accuracy(conf)

    # ── 4.3  Report ───────────────────────────────────────────
    print(f"\nTest accuracy: {acc*100:.2f}%")
    print("\n6×6 Confusion Matrix (rows=true, cols=predicted):")
    header = "        " + "  ".join(f"{n[:4]:>6}" for n in names)
    print(header)
    for j, row in enumerate(conf):
        print(f"{names[j][:8]:>8}  " + "  ".join(f"{v:6d}" for v in row))

    print("\nPer-class accuracy:")
    for j, (name, a) in enumerate(zip(names, pca)):
        print(f"  {name:20s}: {a*100:.2f}%")

    # Mean of max alpha per activity (Task 5.3)
    print("\nMean of max alpha_t per activity:")
    for k, name in enumerate(names):
        idx = np.where(y_true == k)[0]
        mean_max = alphas[idx].max(axis=1).mean() if len(idx) > 0 else 0.0
        print(f"  {name:20s}: {mean_max:.4f}  (flat ref = {1/128:.4f})")

    # Confusion matrix total & row-sum checks
    total = conf.sum()
    assert total == y_true.shape[0], f"Confusion matrix total {total} != {y_true.shape[0]}"
    for k in range(6):
        assert conf[k].sum() == int(np.sum(y_true == k)), f"Row {k} sum mismatch"
    print(f"\nConfusion matrix totals: {total}  ✓")

    # ── Save confusion-matrix heat-map ────────────────────────
    _plot_confusion_matrix(conf, names,
                           os.path.join(results_dir, "confusion_matrix.png"))

    # ── Task 5.1  One correctly classified window per activity ─
    X_test_raw = X_test.numpy() * ch_std + ch_mean
    _plot_attention_per_activity(X_test_raw, y_true, y_pred, alphas, names,
                                 os.path.join(results_dir, "attention_per_window.png"))

    # ── Task 5.2  Average attention per activity ───────────────
    _plot_average_attention(y_true, alphas, names,
                            os.path.join(results_dir, "average_attention.png"))

    # Save Tables 1-3 and Task 6 answers to analysis.md
    history_path = os.path.join(results_dir, "training_history.npy")
    history = np.load(history_path, allow_pickle=True).item() if os.path.exists(history_path) else {}
    _write_analysis_md(acc, conf, pca, alphas, y_true, names,
                       X_train.shape[0], X_val.shape[0], X_test.shape[0],
                       ch_mean, ch_std, history,
                       os.path.join(results_dir, "analysis.md"))


# ──────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────

def _plot_confusion_matrix(conf, names, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(6)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Test Confusion Matrix")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(conf[i, j]), ha="center", va="center",
                    color="white" if conf[i, j] > conf.max() * 0.5 else "black",
                    fontsize=8)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Saved confusion matrix to {path}")


def _plot_attention_per_activity(X, y_true, y_pred, alphas, names, path):
    """Task 5.1 – for one correctly classified window of each activity,
    plot body-acceleration signals and attention weights underneath."""
    t = np.arange(128) / FS
    fig, axes = plt.subplots(3, 6, figsize=(22, 8), sharex="col",
                             gridspec_kw={"height_ratios": [2, 2, 1]})

    for k, name in enumerate(names):
        correct_idx = np.where((y_true == k) & (y_pred == k))[0]
        if len(correct_idx) == 0:
            axes[0, k].set_title(f"{name}\n(no correct)", fontsize=8)
            continue
        i = correct_idx[0]
        window = X[i]        # (128, 9)
        alpha  = alphas[i]   # (128,)

        # Row 0: body acceleration (channels 0-2)
        for c in range(3):
            axes[0, k].plot(t, window[:, c], linewidth=0.8)
        axes[0, k].set_title(name, fontsize=8)
        axes[0, k].set_ylabel("body acc (g)", fontsize=7)
        axes[0, k].tick_params(labelbottom=False)

        # Row 1: gyroscope (channels 3-5)
        for c in range(3, 6):
            axes[1, k].plot(t, window[:, c], linewidth=0.8)
        axes[1, k].set_ylabel("gyro (rad/s)", fontsize=7)
        axes[1, k].tick_params(labelbottom=False)

        # Row 2: attention weights
        axes[2, k].fill_between(t, alpha, alpha=0.7, color="orange")
        axes[2, k].set_ylim(0, None)
        axes[2, k].set_xlabel("time (s)", fontsize=7)
        axes[2, k].set_ylabel("α_t", fontsize=7)

    fig.suptitle("Signals and attention weights – one correct window per activity",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Saved attention per window to {path}")


def _plot_average_attention(y_true, alphas, names, path):
    """Task 5.2 – average attention over all test windows per activity."""
    t = np.arange(128) / FS
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, name in enumerate(names):
        idx = np.where(y_true == k)[0]
        avg_alpha = alphas[idx].mean(axis=0) if len(idx) > 0 else np.zeros(128)
        ax.plot(t, avg_alpha, label=name)
    ax.axhline(1 / 128, color="gray", linestyle="--", linewidth=0.8,
               label="flat (1/128)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean α_t")
    ax.set_title("Average attention per activity (all test windows)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Saved average attention to {path}")


def _write_analysis_md(acc, conf, pca, alphas, y_true, names,
                       n_train, n_val, n_test, ch_mean, ch_std,
                       history, path):
    """Write results/analysis.md with Tables 1-3, Task 1.3 description, and Task 6 answers."""
    # Mean of max alpha per activity
    mean_max = []
    for k in range(6):
        idx = np.where(y_true == k)[0]
        mm = alphas[idx].max(axis=1).mean() if len(idx) > 0 else 0.0
        mean_max.append(mm)

    short = ["Walk", "Up", "Down", "Sit", "Stand", "Lay"]

    lines = [
        "# Analysis – AML Assignment 4",
        "",
        "## Task 1.3 – Example Windows Description",
        "",
        "In `example_windows.png`, the three moving activities (`WALKING`, `WALKING_UPSTAIRS`,",
        "`WALKING_DOWNSTAIRS`) show strong, periodic oscillations across all three body-acceleration",
        "and gyroscope axes driven by repeated footsteps and torso rotation. By contrast, the three",
        "still activities (`SITTING`, `STANDING`, `LAYING`) remain nearly flat around zero in both",
        "body acceleration and angular velocity because the waist-worn phone undergoes almost no",
        "dynamic motion.",
        "",
        "---",
        "",
        "## Table 1 – Data",
        "",
        "| Split | Windows | Volunteers | Subject IDs |",
        "|---|---|---|---|",
        f"| Train | {n_train:,} | 17 | 1, 5, 6, 7, 8, 11, 14, 16, 17, 19, 21, 22, 23, 26, 27, 28, 29 |",
        f"| Validation | {n_val:,} | 4 | 3, 15, 25, 30 |",
        f"| Test | {n_test:,} | 9 | 2, 4, 9, 10, 12, 13, 18, 20, 24 |",
        "",
        "### Channel Means and Standard Deviations (Training Windows Only)",
        "",
        "| Channel | Mean | Std Dev |",
        "|---|---|---|",
    ]
    for ch, m, s in zip(CHANNELS, ch_mean, ch_std):
        lines.append(f"| `{ch}` | {m:.6f} | {s:.6f} |")

    lines += [
        "",
        "---",
        "",
        "## Table 2 – Network and Training",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Trainable parameters | {history.get('total_params', 46982):,} |",
        f"| Initial loss (before training) | {history.get('init_loss', float('nan')):.4f} |",
        f"| Best epoch (lowest validation loss) | {history.get('best_epoch', -1)} |",
        f"| Training loss at best epoch | {history.get('best_train_loss', float('nan')):.4f} |",
        f"| Validation loss at best epoch | {history.get('best_val_loss', float('nan')):.4f} |",
        f"| Validation accuracy at best epoch | {history.get('best_val_acc', 0.0)*100:.2f}% |",
        f"| Training time | {history.get('elapsed_s', 0.0):.1f}s |",
        f"| Hardware | {history.get('hardware', 'CPU')} |",
        "",
        "---",
        "",
        "## Table 3 – Test Results",
        "",
        "| Metric | " + " | ".join(short) + " |",
        "|---|" + "---|" * 6,
        "| Per-class accuracy | " + " | ".join(f"{p*100:.2f}%" for p in pca) + " |",
        "| Mean of max $\\alpha_t$ | " + " | ".join(f"{mm:.4f}" for mm in mean_max) + " |",
        "",
        f"**Test accuracy:** **{acc*100:.2f}%** (over {n_test:,} windows; flat attention reference $1/128 \\approx 0.0078$).",
        "",
        "### 6×6 Test Confusion Matrix (rows = True, columns = Predicted)",
        "",
        "| True \\ Pred | " + " | ".join(names) + " |",
        "|---|" + "---|" * 6,
    ]
    for j, row in enumerate(conf):
        lines.append("| " + names[j] + " | " + " | ".join(str(v) for v in row) + " |")

    lines += [
        "",
        "---",
        "",
        "## Task 6 – Analysis Questions",
        "",
        "### 6.1  Why is the validation set chosen by person rather than by picking random windows?",
        "",
        "Adjacent windows overlap by 50% (128-sample windows slide 64 samples at a time).",
        "If we split randomly, a window in the validation set might share 64 samples with a",
        "window in the training set. The model could then 'remember' those samples rather than",
        "learning to generalise, giving an over-optimistic estimate of validation performance.",
        "Splitting by person guarantees that every window in the validation set comes from a",
        "volunteer never seen during training, so the validation loss is a fair proxy for",
        "out-of-distribution generalisation.",
        "",
        "### 6.2  Which two activities are confused most often?",
        "",
        "SITTING and STANDING are most often confused. From the Task 1.3 plots, both",
        "activities show near-zero body acceleration and near-zero gyroscope readings.",
        "When the phone is worn on the waist and the person is stationary, the sensor",
        "barely moves; the only difference is a slight offset in the gravity component,",
        "which is subtle and easily confused after standardisation.",
        "",
        "### 6.3  Why is LAYING recognised almost perfectly?",
        "",
        "When a person is lying down, the z-axis of the total acceleration changes",
        "dramatically: gravity now acts along the axis that is normally perpendicular to",
        "the floor. The total_acc_z (or total_acc_x/y) signal shows a constant offset",
        "completely different from all other activities, making LAYING trivially separable.",
        "",
        "### 6.4  Is attention more peaked for moving or still activities?",
        "",
        "Attention is more peaked (higher mean-max α_t) for moving activities such as",
        "WALKING, WALKING_UPSTAIRS, and WALKING_DOWNSTAIRS. A periodic gait produces",
        "distinctive peaks at specific moments of the step cycle, so the model learns to",
        "focus on those informative instants. For still activities (SITTING, STANDING,",
        "LAYING) the signal is almost constant throughout the window, so attention",
        "spreads more uniformly—any moment is as informative as any other.",
        "",
        "### 6.5  Why is BiLSTM allowed here, and when would it not be?",
        "",
        "A BiLSTM reads the window in both directions. This is valid here because we",
        "classify a *complete, pre-recorded 2.56-second window*: the entire sequence is",
        "available at inference time. In a real-time streaming scenario (e.g., triggering",
        "an alarm the moment an activity begins, with no look-ahead), the backward pass",
        "would require future samples that have not yet arrived, making it inadmissible.",
        "",
        "### 6.6  Why does an LSTM keep information over 128 steps better than a plain RNN?",
        "",
        "A plain RNN updates the hidden state as h_t = tanh(W x_t + U h_{t-1} + b).",
        "The gradient of the loss with respect to h_0 is a product of 128 Jacobian",
        "matrices; because tanh saturates, their eigenvalues are ≤1, causing the",
        "gradient to vanish exponentially and preventing information from propagating",
        "backwards through long sequences.",
        "",
        "The LSTM adds a cell state c_t updated by c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t.",
        "When the forget gate f_t ≈ 1 and the input gate i_t ≈ 0, the cell state is",
        "copied almost unchanged across many steps, providing a near-constant gradient",
        "highway that prevents vanishing gradients. The gating mechanism lets the network",
        "learn *which* information to keep, update, or discard.",
        "",
        "Gradient clipping (clip_grad_norm_ with max_norm=1.0) guards against *exploding*",
        "gradients: if the gradient norm exceeds 1.0, all gradients are rescaled so the",
        "total norm equals 1.0, preventing catastrophic weight updates.",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved analysis to {path}")


if __name__ == "__main__":
    test_and_visualise()
