"""
BT3068 — Class Weight Computation
===================================
Handles class imbalance by computing inverse-frequency class weights.
Penalizes minority-class errors more heavily during training.
"""

import os
import numpy as np
from collections import Counter


def compute_class_weights(data_dir):
    """
    Compute class weights from directory structure.

    For a dataset with class distribution:
      Pneumonia: 3875 (62.5%)
      Normal:    1341 (37.5%)

    Class weights will penalize under-represented classes more:
      w_i = total_samples / (n_classes × class_count_i)

    Args:
        data_dir: Path to training data directory with subdirectories per class

    Returns:
        class_weight_dict: {class_index: weight} for model.fit()
        class_names: list of class names (sorted alphabetically)
    """
    class_names = sorted([d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d))])

    class_counts = {}
    for i, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        count = len([f for f in os.listdir(class_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
        class_counts[i] = count

    total_samples = sum(class_counts.values())
    n_classes = len(class_names)

    # Inverse frequency weighting
    class_weights = {}
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = total_samples / (n_classes * count)

    # Print summary
    print("=" * 60)
    print("CLASS WEIGHT COMPUTATION")
    print("=" * 60)
    for i, name in enumerate(class_names):
        print(f"  [{i}] {name:20s} → {class_counts[i]:6d} samples → weight = {class_weights[i]:.4f}")
    print(f"  {'TOTAL':25s} → {total_samples:6d} samples")
    print("=" * 60)

    return class_weights, class_names


def compute_class_weights_from_labels(labels):
    """
    Compute class weights from a label array (for non-directory datasets).

    Args:
        labels: numpy array of integer class labels

    Returns:
        class_weight_dict: {class_index: weight}
    """
    counter = Counter(labels)
    total = len(labels)
    n_classes = len(counter)

    class_weights = {}
    for cls, count in counter.items():
        class_weights[cls] = total / (n_classes * count)

    return class_weights


def compute_sample_weights(labels, class_weights):
    """
    Compute per-sample weights for datasets that don't support class_weight.

    Args:
        labels: array of integer labels
        class_weights: dict from compute_class_weights

    Returns:
        sample_weights: numpy array of per-sample weights
    """
    sample_weights = np.array([class_weights[label] for label in labels])
    return sample_weights


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "../chest_xray/train"

    if os.path.exists(data_dir):
        weights, names = compute_class_weights(data_dir)
        print(f"\nClass weights ready for model.fit(class_weight={weights})")
    else:
        print(f"[ERROR] Directory not found: {data_dir}")
        print("[INFO] Usage: python class_weights.py <path_to_train_dir>")
