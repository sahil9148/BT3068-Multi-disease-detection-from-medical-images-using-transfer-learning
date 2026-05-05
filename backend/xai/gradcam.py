"""
BT3068 — Grad-CAM (Gradient-weighted Class Activation Maps)
=============================================================
Generates visual heatmap overlays showing which image regions
triggered the model's disease prediction.

Reference: Selvaraju et al. (2017). "Grad-CAM: Visual Explanations
from Deep Networks via Gradient-based Localization."
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


def get_gradcam_heatmap(model, image, pred_class=None, last_conv_layer_name=None):
    """
    Generate Grad-CAM heatmap for a given image and prediction class.

    Args:
        model: Trained Keras model
        image: Input image tensor (H, W, 3), normalized to [0, 1]
        pred_class: Target class index (default: predicted class)
        last_conv_layer_name: Name of last conv layer (auto-detected if None)

    Returns:
        heatmap: Numpy array (H, W) with activation intensities [0, 1]
    """
    # Auto-detect last conv layer
    if last_conv_layer_name is None:
        last_conv_layer_name = _find_last_conv_layer(model)

    # Build gradient model
    grad_model = keras.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Add batch dimension
    if len(image.shape) == 3:
        image_batch = tf.expand_dims(image, 0)
    else:
        image_batch = image

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        if pred_class is None:
            pred_class = tf.argmax(predictions[0])
        class_output = predictions[:, pred_class]

    # Gradient of the predicted class w.r.t. conv layer output
    grads = tape.gradient(class_output, conv_outputs)

    # Global Average Pooling of gradients → channel importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv output channels by their importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU + normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    if tf.reduce_max(heatmap) > 0:
        heatmap = heatmap / tf.reduce_max(heatmap)

    return heatmap.numpy()


def _find_last_conv_layer(model):
    """Auto-detect the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):
            # Nested model (e.g., EfficientNetB4 inside functional model)
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
                    return sublayer.name
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
            return layer.name
    raise ValueError("No convolutional layer found in model.")


def generate_gradcam_for_batch(model, images, pred_classes=None):
    """
    Generate Grad-CAM heatmaps for a batch of images.

    Args:
        model: Trained model
        images: Batch of images (B, H, W, 3)
        pred_classes: Optional list of target classes

    Returns:
        List of heatmap arrays
    """
    heatmaps = []
    for i in range(len(images)):
        pc = pred_classes[i] if pred_classes is not None else None
        heatmap = get_gradcam_heatmap(model, images[i], pred_class=pc)
        heatmaps.append(heatmap)
    return heatmaps


if __name__ == "__main__":
    print("[Grad-CAM] Module loaded. Use get_gradcam_heatmap(model, image) to generate heatmaps.")
