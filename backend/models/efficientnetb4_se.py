"""
BT3068 — EfficientNetB4 + Squeeze-and-Excitation (SE) Channel Attention
=========================================================================
Primary backbone model for multi-disease detection.

Architecture:
  EfficientNetB4 (ImageNet) → SE Block → GAP → Dense → Classification

SE Block (Hu et al., 2018 — CVPR):
  1. Global Average Pooling → C-dim descriptor
  2. Dense(C/16) → ReLU → Dense(C) → Sigmoid
  3. Element-wise multiply → re-calibrated feature maps
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


class SqueezeExcitationBlock(layers.Layer):
    """
    Squeeze-and-Excitation Channel Attention Block.

    Learns per-channel scaling weights to re-calibrate feature maps,
    forcing the network to focus on pathology-relevant image regions.

    Reference: Hu, Shen & Sun (2018). "Squeeze-and-Excitation Networks." CVPR.
    """

    def __init__(self, reduction_ratio=16, **kwargs):
        super(SqueezeExcitationBlock, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced_channels = max(channels // self.reduction_ratio, 1)

        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense_squeeze = layers.Dense(
            reduced_channels,
            activation='relu',
            kernel_initializer='he_normal',
            name='se_squeeze'
        )
        self.dense_excite = layers.Dense(
            channels,
            activation='sigmoid',
            kernel_initializer='he_normal',
            name='se_excite'
        )
        self.reshape = layers.Reshape((1, 1, channels))

    def call(self, inputs):
        # Squeeze: Global Average Pooling → C-dimensional descriptor
        se = self.global_avg_pool(inputs)

        # Excitation: FC → ReLU → FC → Sigmoid
        se = self.dense_squeeze(se)
        se = self.dense_excite(se)

        # Reshape for element-wise multiplication
        se = self.reshape(se)

        # Scale: element-wise multiply to re-calibrate channels
        return inputs * se

    def get_config(self):
        config = super().get_config()
        config.update({'reduction_ratio': self.reduction_ratio})
        return config


def build_efficientnetb4_se(num_classes, input_shape=(380, 380, 3),
                              reduction_ratio=16, dropout_rate=0.4,
                              freeze_base=True):
    """
    Build EfficientNetB4 + SE Channel Attention model.

    Architecture:
        Input (380×380×3)
        → EfficientNetB4 backbone (ImageNet pre-trained)
        → Squeeze-and-Excitation Block (channel re-calibration)
        → Global Average Pooling
        → Dropout (0.4)
        → Dense(256, ReLU)
        → Dropout (0.3)
        → Dense(num_classes, Softmax)

    Args:
        num_classes: Number of output disease categories
        input_shape: Input image dimensions (default 380×380 for B4)
        reduction_ratio: SE block channel reduction ratio (default 16)
        dropout_rate: Dropout rate before classification head
        freeze_base: If True, freeze EfficientNetB4 backbone layers

    Returns:
        Keras Model ready for compilation
    """
    # ═════════ Backbone ═════════
    base_model = keras.applications.EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze/unfreeze backbone
    base_model.trainable = not freeze_base

    # ═════════ Model Architecture ═════════
    inputs = keras.Input(shape=input_shape, name='medical_image_input')

    # EfficientNetB4 feature extraction
    x = base_model(inputs, training=not freeze_base)

    # Squeeze-and-Excitation Channel Attention
    x = SqueezeExcitationBlock(
        reduction_ratio=reduction_ratio,
        name='channel_attention'
    )(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)

    # Classification Head
    x = layers.Dropout(dropout_rate, name='dropout_1')(x)
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal',
                     name='dense_hidden')(x)
    x = layers.BatchNormalization(name='bn_head')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)

    # Output
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer='glorot_uniform',
        name='disease_classification'
    )(x)

    model = Model(inputs=inputs, outputs=outputs,
                  name='EfficientNetB4_SE_MultiDisease')

    return model


def unfreeze_top_layers(model, num_layers=100, base_model_name='efficientnetb4'):
    """
    Unfreeze the top N layers of the backbone for Phase 2 fine-tuning.

    Args:
        model: Compiled Keras model
        num_layers: Number of top backbone layers to unfreeze (default 100)
        base_model_name: Name of the backbone layer

    Returns:
        model with updated trainable status
    """
    # Find the base model layer
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 10:
            base_model = layer
            break

    if base_model is None:
        print("[WARNING] Could not find base model layer. Unfreezing all layers.")
        model.trainable = True
        return model

    # Unfreeze top N layers
    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = max(0, total_layers - num_layers)

    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    trainable = sum(1 for l in base_model.layers if l.trainable)
    frozen = total_layers - trainable

    print(f"[FINE-TUNE] Base model: {total_layers} total layers")
    print(f"[FINE-TUNE] Frozen: {frozen} | Trainable: {trainable}")
    print(f"[FINE-TUNE] Top {num_layers} layers unfrozen for fine-tuning")

    return model


def get_model_summary(num_classes=2):
    """Print model summary for inspection."""
    model = build_efficientnetb4_se(num_classes=num_classes, freeze_base=True)
    model.summary()

    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable = total_params - trainable_params

    print(f"\n{'=' * 60}")
    print(f"MODEL: EfficientNetB4 + SE Channel Attention")
    print(f"{'=' * 60}")
    print(f"  Total Parameters:         {total_params:>12,}")
    print(f"  Trainable Parameters:     {trainable_params:>12,}")
    print(f"  Non-trainable Parameters: {non_trainable:>12,}")
    print(f"  Output Classes:           {num_classes:>12}")
    print(f"{'=' * 60}")

    return model


if __name__ == "__main__":
    print("Building EfficientNetB4 + SE model for Chest X-Ray (2 classes)...")
    model = get_model_summary(num_classes=2)
