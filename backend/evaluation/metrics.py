"""
BT3068 — Evaluation Metrics
==============================
Computes Accuracy, AUROC, F1-Score, Cohen's Kappa, Sensitivity, Specificity.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, cohen_kappa_score, classification_report,
    confusion_matrix
)


def compute_all_metrics(y_true, y_pred_proba, class_names=None):
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels (integer encoded)
        y_pred_proba: Predicted probability matrix (n_samples, n_classes)
        class_names: Optional list of class names

    Returns:
        Dictionary of all computed metrics
    """
    y_pred = np.argmax(y_pred_proba, axis=1)
    n_classes = y_pred_proba.shape[1]

    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')

    # Cohen's Kappa
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

    # AUC-ROC
    if n_classes == 2:
        metrics['auroc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
    else:
        try:
            metrics['auroc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        except ValueError:
            metrics['auroc'] = None

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    # Per-class metrics
    per_class = {}
    for i in range(n_classes):
        name = class_names[i] if class_names else f"Class_{i}"
        mask = (y_true == i)
        per_class[name] = {
            'precision': precision_score(y_true, y_pred, labels=[i], average='micro'),
            'recall': recall_score(y_true, y_pred, labels=[i], average='micro'),
            'f1': f1_score(y_true, y_pred, labels=[i], average='micro'),
            'support': int(np.sum(mask))
        }
    metrics['per_class'] = per_class

    # Sensitivity & Specificity (binary)
    if n_classes == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return metrics


def print_metrics_report(metrics, title="EVALUATION REPORT"):
    """Pretty-print metrics report."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  F1 (Weighted):   {metrics['f1_weighted']:.4f}")
    print(f"  Precision (W):   {metrics['precision_weighted']:.4f}")
    print(f"  Recall (W):      {metrics['recall_weighted']:.4f}")
    print(f"  Cohen's Kappa:   {metrics['cohen_kappa']:.4f}")
    if metrics.get('auroc'):
        print(f"  AUC-ROC:         {metrics['auroc']:.4f}")
    if metrics.get('sensitivity') is not None:
        print(f"  Sensitivity:     {metrics['sensitivity']:.4f}")
        print(f"  Specificity:     {metrics['specificity']:.4f}")
    print("=" * 60)

    if 'per_class' in metrics:
        print("\n  Per-Class Metrics:")
        for name, vals in metrics['per_class'].items():
            print(f"    {name:20s}  P={vals['precision']:.3f}  R={vals['recall']:.3f}  "
                  f"F1={vals['f1']:.3f}  N={vals['support']}")
    print()
