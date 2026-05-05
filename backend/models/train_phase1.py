"""
BT3068 — Phase 1 Training: Frozen Base Feature Extraction
============================================================
Trains SE block + classification head with EfficientNetB4 backbone FROZEN.
"""

import os
import sys
import argparse
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.efficientnetb4_se import build_efficientnetb4_se
from data.preprocessing.augmentation import create_train_val_pipeline
from data.preprocessing.class_weights import compute_class_weights


def get_callbacks(checkpoint_dir='checkpoints/phase1'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    return [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model_phase1.keras'),
            monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir='logs/phase1', histogram_freq=1),
    ]


def train_phase1(data_dir, num_classes=2, epochs=20, batch_size=32,
                 image_size=(380, 380), learning_rate=1e-4):
    print("=" * 70)
    print("  PHASE 1 — FROZEN BASE FEATURE EXTRACTION")
    print("=" * 70)

    train_ds, val_ds, class_names = create_train_val_pipeline(data_dir, image_size=image_size, batch_size=batch_size)
    num_classes = len(class_names)

    class_weights, _ = compute_class_weights(data_dir)

    model = build_efficientnetb4_se(num_classes=num_classes, input_shape=(*image_size, 3), freeze_base=True)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        class_weight=class_weights, callbacks=get_callbacks(), verbose=1)

    best_val_acc = max(history.history['val_accuracy'])
    print(f"\n  Phase 1 Complete — Best Val Accuracy: {best_val_acc:.4f}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BT3068 Phase 1 Training')
    parser.add_argument('--dataset', type=str, default='data/chest_xray/train')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_classes', type=int, default=2)
    args = parser.parse_args()
    train_phase1(args.dataset, args.num_classes, args.epochs, args.batch_size, learning_rate=args.lr)
