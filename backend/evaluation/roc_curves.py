"""
BT3068 — ROC Curve Plotting
==============================
Per-class and aggregate ROC curves with AUC values.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_roc_curves(y_true, y_pred_proba, class_names, title="ROC Curves",
                     save_path=None, figsize=(10, 8)):
    """
    Plot per-class ROC curves with AUC values.

    Args:
        y_true: True labels (integer encoded)
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        class_names: List of class names
        title: Plot title
        save_path: Optional save path
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, color='navy', lw=3, linestyle='--',
            label=f'Micro-avg (AUC = {roc_auc_micro:.3f})')

    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.500)')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] ROC curves → {save_path}")
    plt.show()


if __name__ == "__main__":
    # Demo with random data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 200)
    y_proba = np.random.rand(200, 2)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    plot_roc_curves(y_true, y_proba, ['Normal', 'Pneumonia'])
