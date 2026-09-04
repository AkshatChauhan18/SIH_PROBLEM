"""
src/utils/metrics.py
Evaluation metrics: Macro F1, Per-Class F1, Precision, Recall, FPR, Forecast Lead Time.
"""

from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

def compute_evaluation_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Computes comprehensive multi-class evaluation metrics:
    - Macro F1, Weighted F1
    - Per-Class Precision, Recall, F1
    - False Positive Rate (FPR) for Malicious vs Benign
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    per_class_stats = {}
    for idx, name in enumerate(class_names):
        if idx < len(per_class_f1):
            per_class_stats[name] = {
                "f1": float(per_class_f1[idx]),
                "precision": float(per_class_precision[idx]),
                "recall": float(per_class_recall[idx]),
                "support": int(np.sum(y_true == idx)),
            }

    # Binary FPR: Benign (class 0) vs Any Attack (classes > 0)
    # False Positive = True is Benign (0), but predicted Attack (> 0)
    is_true_benign = (y_true == 0)
    is_pred_attack = (y_pred > 0)
    fp = int(np.sum(is_true_benign & is_pred_attack))
    tn = int(np.sum(is_true_benign & ~is_pred_attack))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "false_positive_rate": fpr,
        "per_class": per_class_stats,
    }

def compute_lead_time_gain(
    horizon_step: int,
    window_seconds: int = 30
) -> Dict[str, Any]:
    """
    Computes early warning forecast lead time:
    Lead Time = horizon_step * window_seconds
    e.g. k=1: 30s advance warning, k=5: 150s (2.5 min) advance warning.
    """
    lead_seconds = horizon_step * window_seconds
    return {
        "horizon_step": horizon_step,
        "lead_time_seconds": lead_seconds,
        "lead_time_minutes": round(lead_seconds / 60.0, 2),
    }
