"""
BT3068 — GPU-Accelerated Data Augmentation Pipeline
====================================================
Implements on-GPU augmentation using tf.data pipeline with:
- Random flipping, rotation, zoom, contrast, brightness
- Gaussian blur for noise smoothing
- Cache → Prefetch for maximum throughput
"""

import tensorflow as tf
import numpy as np


# ══════════════════════════════════════════
# Augmentation Layer Stack
# ══════════════════════════════════════════

def build_augmentation_layer():
    """
    Build a Keras Sequential augmentation layer for on-GPU augmentation.
    Applied during training only (layers are no-op during inference).
    """
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(
            factor=0.083,  # ±30° (30/360 ≈ 0.083)
            fill_mode='reflect'
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.2, 0.2),  # Zoom 0.8× to 1.2×
            width_factor=(-0.2, 0.2),
            fill_mode='reflect'
        ),
        tf.keras.layers.RandomContrast(factor=0.2),
        tf.keras.layers.RandomBrightness(factor=0.1),
    ], name='gpu_augmentation')
    return augmentation


def gaussian_blur(image, kernel_size=3, sigma=1.0):
    """
    Apply Gaussian blur for acquisition noise smoothing.
    Uses depthwise convolution for GPU acceleration.
    """
    # Create Gaussian kernel
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
    kernel_1d = tf.exp(-0.5 * (x / sigma) ** 2)
    kernel_2d = tf.tensordot(kernel_1d, kernel_1d, axes=0)
    kernel_2d = kernel_2d / tf.reduce_sum(kernel_2d)

    # Reshape for depthwise convolution: [H, W, in_channels, channel_multiplier]
    channels = tf.shape(image)[-1]
    kernel_2d = tf.reshape(kernel_2d, [kernel_size, kernel_size, 1, 1])
    kernel_2d = tf.tile(kernel_2d, [1, 1, channels, 1])

    # Apply blur
    if len(image.shape) == 3:
        image = tf.expand_dims(image, 0)
        blurred = tf.nn.depthwise_conv2d(image, kernel_2d, strides=[1, 1, 1, 1], padding='SAME')
        return tf.squeeze(blurred, 0)
    else:
        return tf.nn.depthwise_conv2d(image, kernel_2d, strides=[1, 1, 1, 1], padding='SAME')


# ══════════════════════════════════════════
# Dataset Loading & Pipeline
# ══════════════════════════════════════════

def load_dataset_from_directory(data_dir, image_size=(380, 380), batch_size=32,
                                 validation_split=0.2, seed=42):
    """
    Load dataset from directory structure using tf.keras.utils.image_dataset_from_directory.

    Expected directory structure:
    data_dir/
    ├── class_1/
    │   ├── image_001.jpg
    │   └── ...
    └── class_2/
        ├── image_001.jpg
        └── ...

    Returns:
        train_ds, val_ds, class_names
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    class_names = train_ds.class_names
    return train_ds, val_ds, class_names


def build_pipeline(dataset, augment=False, image_size=(380, 380)):
    """
    Build optimized tf.data pipeline with:
    - Normalization to [0, 1]
    - Optional augmentation (training only)
    - Cache → Prefetch for throughput
    """
    AUTOTUNE = tf.data.AUTOTUNE

    # Normalize pixel values to [0, 1]
    normalization = tf.keras.layers.Rescaling(1.0 / 255.0)

    # Build augmentation layer
    augmentation_layer = build_augmentation_layer() if augment else None

    def preprocess(images, labels):
        images = normalization(images)
        if augment and augmentation_layer is not None:
            images = augmentation_layer(images, training=True)
        return images, labels

    dataset = dataset.cache()
    dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset


def create_train_val_pipeline(data_dir, image_size=(380, 380), batch_size=32,
                                validation_split=0.2):
    """
    End-to-end pipeline creation: load → augment → cache → prefetch.

    Args:
        data_dir: Path to dataset directory
        image_size: Target image dimensions (default 380×380 for EfficientNetB4)
        batch_size: Training batch size
        validation_split: Fraction for validation

    Returns:
        train_ds, val_ds, class_names
    """
    train_ds, val_ds, class_names = load_dataset_from_directory(
        data_dir, image_size=image_size, batch_size=batch_size,
        validation_split=validation_split
    )

    # Training: with augmentation
    train_ds = build_pipeline(train_ds, augment=True, image_size=image_size)

    # Validation: no augmentation
    val_ds = build_pipeline(val_ds, augment=False, image_size=image_size)

    print(f"[PIPELINE] Dataset loaded from: {data_dir}")
    print(f"[PIPELINE] Classes: {class_names}")
    print(f"[PIPELINE] Image size: {image_size}")
    print(f"[PIPELINE] Batch size: {batch_size}")
    print(f"[PIPELINE] Augmentation: ON (training) / OFF (validation)")

    return train_ds, val_ds, class_names


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "../chest_xray/train"

    train_ds, val_ds, class_names = create_train_val_pipeline(data_dir)
    print(f"\n[READY] Training pipeline built successfully.")
    print(f"[READY] {len(class_names)} classes: {class_names}")
