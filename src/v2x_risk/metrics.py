from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

CLASS_NAMES = ["low", "medium", "high"]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
    )
    macro_precision, macro_recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average="weighted", zero_division=0
    )
    ordinal_error = y_pred.astype(float) - y_true.astype(float)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "ordinal_mae": float(np.abs(ordinal_error).mean()),
        "ordinal_rmse": float(np.sqrt(np.square(ordinal_error).mean())),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
    }
