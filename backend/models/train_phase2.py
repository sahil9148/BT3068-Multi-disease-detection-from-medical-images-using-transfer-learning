"""
BT3068 — Phase 2 Training: Fine-Tuning Top 100 Layers
========================================================
Unfreezes top 100 EfficientNetB4 layers for domain adaptation.
Uses lower learning rate (1e-5) to avoid catastrophic forgetting.
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


def get_callbacks(checkpoint_dir='checkpoints/phase2'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    return [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-8, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model_phase2.keras'),
            monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir='logs/phase2', histogram_freq=1),
    ]


def train_phase2(data_dir, phase1_model_path=None, num_classes=2, epochs=30,
                 batch_size=32, image_size=(380, 380), learning_rate=1e-5,
                 unfreeze_layers=100):
    print("=" * 70)
    print("  PHASE 2 — FINE-TUNING TOP 100 LAYERS")
    print("=" * 70)

    train_ds, val_ds, class_names = create_train_val_pipeline(data_dir, image_size=image_size, batch_size=batch_size)
    num_classes = len(class_names)
    class_weights, _ = compute_class_weights(data_dir)

    # Load Phase 1 model or build fresh
    if phase1_model_path and os.path.exists(phase1_model_path):
        print(f"\n[LOAD] Loading Phase 1 model from: {phase1_model_path}")
        model = keras.models.load_model(phase1_model_path)
    else:
        print("\n[BUILD] Building fresh model (no Phase 1 checkpoint found)")
        model = build_efficientnetb4_se(num_classes=num_classes, input_shape=(*image_size, 3), freeze_base=True)

    # Unfreeze top layers for fine-tuning
    model = unfreeze_top_layers(model, num_layers=unfreeze_layers)

    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"\n[TRAIN] Fine-tuning with lr={learning_rate}, epochs={epochs}")

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        class_weight=class_weights, callbacks=get_callbacks(), verbose=1)

    best_val_acc = max(history.history['val_accuracy'])
    print(f"\n  Phase 2 Complete — Best Val Accuracy: {best_val_acc:.4f}")
    
    # Save final model
    final_path = 'checkpoints/final_model.keras'
    os.makedirs('checkpoints', exist_ok=True)
    model.save(final_path)
    print(f"  Final model saved to: {final_path}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BT3068 Phase 2 Fine-Tuning')
    parser.add_argument('--dataset', type=str, default='data/chest_xray/train')
    parser.add_argument('--phase1_model', type=str, default='checkpoints/phase1/best_model_phase1.keras')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--unfreeze_layers', type=int, default=100)
    args = parser.parse_args()
    train_phase2(args.dataset, args.phase1_model, epochs=args.epochs,
                 batch_size=args.batch_size, learning_rate=args.lr,
                 unfreeze_layers=args.unfreeze_layers)
