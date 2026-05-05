"""
BT3068 — Confusion Matrix Visualization
==========================================
Generates publication-quality confusion matrix heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix",
                           save_path=None, figsize=(8, 6)):
    """
    Plot confusion matrix as a styled heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure dimensions
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title(f'{title} — Counts', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1],
                vmin=0, vmax=1, cbar_kws={'label': 'Rate'})
    axes[1].set_title(f'{title} — Normalized', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] Confusion matrix → {save_path}")
    plt.show()

    return cm


def print_confusion_matrix(y_true, y_pred, class_names):
    """Print confusion matrix to console."""
    cm = confusion_matrix(y_true, y_pred)
    print("\n  CONFUSION MATRIX")
    print("  " + "-" * 40)
    header = "  {:>15s}".format("") + "".join(f"  Pred:{n:>8s}" for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = f"  True:{name:>8s}" + "".join(f"  {cm[i][j]:>12d}" for j in range(len(class_names)))
        print(row)
    print()
