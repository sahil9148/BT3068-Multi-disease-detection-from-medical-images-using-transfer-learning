"""
BT3068 — DeepLabv3+ Lung Region Segmentation
================================================
Isolates lung regions from chest X-rays using DeepLabv3+ semantic segmentation.
Removes background noise (ribs, heart shadow, diaphragm) for cleaner classification.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


def build_deeplabv3_plus(input_shape=(380, 380, 3), num_classes=2):
    """
    Build a simplified DeepLabv3+ model for lung region isolation.

    Architecture:
    - Encoder: ResNet50 backbone (pre-trained on ImageNet)
    - ASPP: Atrous Spatial Pyramid Pooling (rates = 6, 12, 18)
    - Decoder: Bilinear upsampling + skip connection from encoder

    Args:
        input_shape: Input image dimensions
        num_classes: 2 (lung / non-lung binary segmentation)

    Returns:
        Keras Model for semantic segmentation
    """
    # Encoder — ResNet50 backbone
    backbone = keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Extract feature maps at different scales
    # Low-level features for decoder skip connection
    low_level_features = backbone.get_layer('conv2_block3_out').output  # 1/4 scale
    # High-level features for ASPP
    high_level_features = backbone.output  # 1/32 scale

    # ═════════ ASPP Module ═════════
    aspp_outputs = []

    # 1×1 convolution
    x1 = keras.layers.Conv2D(256, 1, padding='same', use_bias=False)(high_level_features)
    x1 = keras.layers.BatchNormalization()(x1)
    x1 = keras.layers.ReLU()(x1)
    aspp_outputs.append(x1)

    # Atrous convolutions at different rates
    for rate in [6, 12, 18]:
        x = keras.layers.Conv2D(256, 3, padding='same', dilation_rate=rate,
                                 use_bias=False)(high_level_features)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        aspp_outputs.append(x)

    # Global Average Pooling branch
    x_pool = keras.layers.GlobalAveragePooling2D()(high_level_features)
    x_pool = keras.layers.Reshape((1, 1, -1))(x_pool)
    x_pool = keras.layers.Conv2D(256, 1, padding='same', use_bias=False)(x_pool)
    x_pool = keras.layers.BatchNormalization()(x_pool)
    x_pool = keras.layers.ReLU()(x_pool)
    target_size = tf.shape(high_level_features)[1:3]
    x_pool = keras.layers.Lambda(
        lambda x: tf.image.resize(x, target_size)
    )(x_pool)
    aspp_outputs.append(x_pool)

    # Concatenate ASPP outputs
    aspp_concat = keras.layers.Concatenate()(aspp_outputs)
    aspp_out = keras.layers.Conv2D(256, 1, padding='same', use_bias=False)(aspp_concat)
    aspp_out = keras.layers.BatchNormalization()(aspp_out)
    aspp_out = keras.layers.ReLU()(aspp_out)

    # ═════════ Decoder ═════════
    # Upsample ASPP output to match low-level features
    decoder_input = keras.layers.UpSampling2D(
        size=(input_shape[0] // (4 * tf.shape(aspp_out)[1]), 
              input_shape[1] // (4 * tf.shape(aspp_out)[2])),
        interpolation='bilinear'
    )(aspp_out)

    # Process low-level features
    low_level = keras.layers.Conv2D(48, 1, padding='same', use_bias=False)(low_level_features)
    low_level = keras.layers.BatchNormalization()(low_level)
    low_level = keras.layers.ReLU()(low_level)

    # Resize decoder_input to match low_level spatial dimensions
    low_level_shape = tf.shape(low_level)[1:3]
    decoder_input = keras.layers.Lambda(
        lambda x: tf.image.resize(x, low_level_shape)
    )(aspp_out)

    # Concatenate
    decoder = keras.layers.Concatenate()([decoder_input, low_level])
    decoder = keras.layers.Conv2D(256, 3, padding='same', use_bias=False)(decoder)
    decoder = keras.layers.BatchNormalization()(decoder)
    decoder = keras.layers.ReLU()(decoder)
    decoder = keras.layers.Conv2D(256, 3, padding='same', use_bias=False)(decoder)
    decoder = keras.layers.BatchNormalization()(decoder)
    decoder = keras.layers.ReLU()(decoder)

    # Final upsampling to original resolution
    decoder = keras.layers.UpSampling2D(size=4, interpolation='bilinear')(decoder)

    # Classification head
    outputs = keras.layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(decoder)

    model = keras.Model(inputs=backbone.input, outputs=outputs, name='DeepLabv3Plus_LungSeg')
    return model


def segment_lung_region(image, model, threshold=0.5):
    """
    Segment lung region from a chest X-ray image.

    Args:
        image: Input image tensor (H, W, 3), normalized to [0, 1]
        model: Trained DeepLabv3+ segmentation model
        threshold: Confidence threshold for lung mask

    Returns:
        masked_image: Image with non-lung regions zeroed out
        lung_mask: Binary mask of lung regions
    """
    # Add batch dimension
    if len(image.shape) == 3:
        image_batch = tf.expand_dims(image, 0)
    else:
        image_batch = image

    # Predict segmentation mask
    prediction = model.predict(image_batch, verbose=0)

    # Extract lung channel (class 1) and threshold
    lung_prob = prediction[0, :, :, 1]  # Lung class probability
    lung_mask = (lung_prob > threshold).astype(np.float32)

    # Resize mask to match original image if needed
    if lung_mask.shape[:2] != image.shape[:2]:
        lung_mask = tf.image.resize(
            lung_mask[..., np.newaxis],
            image.shape[:2]
        ).numpy()[:, :, 0]

    # Apply mask
    masked_image = image * lung_mask[..., np.newaxis]

    return masked_image, lung_mask


def apply_lung_isolation(dataset, model, threshold=0.5):
    """
    Apply lung region isolation to an entire tf.data dataset.

    Args:
        dataset: tf.data.Dataset of (image, label) pairs
        model: Trained DeepLabv3+ model
        threshold: Segmentation confidence threshold

    Returns:
        Processed dataset with lung-isolated images
    """
    def isolate_fn(image, label):
        masked, _ = segment_lung_region(image.numpy(), model, threshold)
        return tf.constant(masked, dtype=tf.float32), label

    processed = dataset.map(
        lambda img, lbl: tf.py_function(
            isolate_fn, [img, lbl], [tf.float32, tf.float32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return processed


if __name__ == "__main__":
    print("[DeepLabv3+] Building lung segmentation model...")
    model = build_deeplabv3_plus()
    model.summary()
    print(f"\n[DeepLabv3+] Model built: {model.count_params():,} parameters")
    print("[DeepLabv3+] Ready for lung region isolation from chest X-rays")
