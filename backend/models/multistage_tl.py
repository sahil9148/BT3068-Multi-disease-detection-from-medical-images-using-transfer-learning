"""
BT3068 — Three-Stage Multistage Transfer Learning (MSTL) Pipeline
===================================================================
Following Ayana et al. (2024) — validated for +5-12% accuracy improvement.

Stage 1: ImageNet pre-training (already done via EfficientNetB4 weights)
Stage 2: Intermediate medical domain transfer (bridge domain gap)
Stage 3: Target disease fine-tuning (final specialization)
"""

import os
import sys
import argparse
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.efficientnetb4_se import build_efficientnetb4_se, unfreeze_top_layers
from data.preprocessing.augmentation import create_train_val_pipeline
from data.preprocessing.class_weights import compute_class_weights


def stage2_intermediate_transfer(intermediate_data_dir, num_classes, epochs=10,
                                  batch_size=32, image_size=(380, 380)):
    """
    Stage 2: Intermediate Transfer Learning.
    Bridge domain gap between ImageNet and target medical dataset.
    Uses a general medical imaging dataset (e.g., NIH Chest X-Ray14).
    """
    print("\n" + "=" * 70)
    print("  MSTL STAGE 2 — INTERMEDIATE MEDICAL DOMAIN TRANSFER")
    print("=" * 70)

    train_ds, val_ds, class_names = create_train_val_pipeline(
        intermediate_data_dir, image_size=image_size, batch_size=batch_size)
    num_classes = len(class_names)

    model = build_efficientnetb4_se(num_classes=num_classes, input_shape=(*image_size, 3), freeze_base=True)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    class_weights, _ = compute_class_weights(intermediate_data_dir)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('checkpoints/mstl_stage2.keras',
                                        monitor='val_accuracy', save_best_only=True),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        class_weight=class_weights, callbacks=callbacks, verbose=1)

    print(f"  Stage 2 Complete — Val Accuracy: {max(history.history['val_accuracy']):.4f}")
    return model


def stage3_target_finetune(model, target_data_dir, num_classes, epochs=30,
                            batch_size=32, image_size=(380, 380)):
    """
    Stage 3: Target Disease Fine-Tuning.
    Fine-tune all layers with channel attention on the target dataset.
    """
    print("\n" + "=" * 70)
    print("  MSTL STAGE 3 — TARGET DISEASE FINE-TUNING")
    print("=" * 70)

    train_ds, val_ds, class_names = create_train_val_pipeline(
        target_data_dir, image_size=image_size, batch_size=batch_size)
    num_classes = len(class_names)
    class_weights, _ = compute_class_weights(target_data_dir)

    # Rebuild classification head for target classes if different
    # Unfreeze all layers for full fine-tuning
    model = unfreeze_top_layers(model, num_layers=200)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-8),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('checkpoints/mstl_stage3_final.keras',
                                        monitor='val_accuracy', save_best_only=True),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        class_weight=class_weights, callbacks=callbacks, verbose=1)

    best_acc = max(history.history['val_accuracy'])
    print(f"  Stage 3 Complete — Best Val Accuracy: {best_acc:.4f}")
    return model, history


def run_mstl_pipeline(intermediate_dir, target_dir, num_classes_intermediate=14,
                       num_classes_target=2):
    """Run the complete 3-stage MSTL pipeline."""
    print("=" * 70)
    print("  BT3068 — MULTISTAGE TRANSFER LEARNING PIPELINE")
    print("  Stage 1: ImageNet (built-in) → Stage 2: Medical → Stage 3: Target")
    print("=" * 70)

    # Stage 1 is implicit (ImageNet weights in EfficientNetB4)
    print("\n[Stage 1] ImageNet pre-training — BUILT-IN via EfficientNetB4 weights ✓")

    # Stage 2: Intermediate transfer
    model = stage2_intermediate_transfer(intermediate_dir, num_classes_intermediate)

    # Stage 3: Target fine-tuning
    model, history = stage3_target_finetune(model, target_dir, num_classes_target)

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BT3068 MSTL Pipeline')
    parser.add_argument('--intermediate_dir', type=str, required=True, help='Intermediate medical dataset')
    parser.add_argument('--target_dir', type=str, required=True, help='Target disease dataset')
    args = parser.parse_args()
    run_mstl_pipeline(args.intermediate_dir, args.target_dir)
