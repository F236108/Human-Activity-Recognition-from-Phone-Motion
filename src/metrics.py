"""
Task 4.2 – Metric functions (no scikit-learn allowed).

Three functions:
  accuracy(y_true, y_pred)        -> float
  confusion_matrix(y_true, y_pred, num_classes) -> (C, C) int array
  per_class_accuracy(conf_matrix) -> (C,) float array
"""

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Overall accuracy: fraction of windows correctly classified.

    Parameters
    ----------
    y_true : (m,) int  – ground-truth labels (0-indexed)
    y_pred : (m,) int  – predicted labels

    Returns
    -------
    float in [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     num_classes: int = 6) -> np.ndarray:
    """
    Confusion matrix C where C[j, k] = number of windows whose
    true activity is j and whose predicted activity is k.
    Rows = true activities, Columns = predicted activities.

    Parameters
    ----------
    y_true      : (m,) int
    y_pred      : (m,) int
    num_classes : int (default 6)

    Returns
    -------
    C : (num_classes, num_classes) int array
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    C = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1
    return C


def per_class_accuracy(conf: np.ndarray) -> np.ndarray:
    """
    Per-class accuracy: for activity j,  C[j,j] / sum_k C[j,k].

    Parameters
    ----------
    conf : (C, C) int array – confusion matrix from confusion_matrix()

    Returns
    -------
    acc : (C,) float array
    """
    conf = np.asarray(conf, dtype=float)
    row_sums = conf.sum(axis=1)                   # (C,)
    # Avoid division by zero for classes with no test windows
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    acc = conf.diagonal() / row_sums_safe
    return acc


# ──────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Perfect predictions
    yt = np.array([0, 1, 2, 0, 1])
    yp = np.array([0, 1, 2, 0, 1])
    assert accuracy(yt, yp) == 1.0

    # 60 % accuracy
    yp2 = np.array([0, 2, 2, 0, 0])
    assert abs(accuracy(yt, yp2) - 0.6) < 1e-9

    # Confusion matrix
    C = confusion_matrix(yt, yp2, num_classes=3)
    print("Confusion matrix:\n", C)
    # C[0,0]=2, C[1,2]=1, C[2,2]=1, C[1,0]=1  → per-class = [1, 0, 1]
    pca = per_class_accuracy(C)
    print("Per-class accuracy:", pca)
    print("All metric self-tests passed ✓")
