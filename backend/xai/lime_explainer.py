"""
BT3068 — LIME Explainer
==========================
Local Interpretable Model-agnostic Explanations for medical images.
Shows feature importance per pixel region (superpixel).
"""

import numpy as np

try:
    from lime import lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("[WARNING] LIME not installed. Run: pip install lime")


def explain_with_lime(model, image, num_samples=1000, num_features=10,
                       top_labels=3, hide_rest=False):
    """
    Generate LIME explanation for a medical image prediction.

    Args:
        model: Trained model with predict method
        image: Input image (H, W, 3), values in [0, 255] or [0, 1]
        num_samples: Number of perturbation samples
        num_features: Number of superpixels to highlight
        top_labels: Number of top predictions to explain
        hide_rest: If True, grey out non-contributing regions

    Returns:
        explanation: LIME explanation object
        segments: Superpixel segmentation mask
        image_with_explanation: Marked-up image array
    """
    if not LIME_AVAILABLE:
        raise ImportError("LIME not available. Install with: pip install lime")

    explainer = lime_image.LimeImageExplainer()

    # Ensure image is in [0, 1] range
    if image.max() > 1.0:
        image = image / 255.0

    # Prediction function wrapper
    def predict_fn(images):
        images = np.array(images, dtype=np.float32)
        if images.max() > 1.0:
            images = images / 255.0
        return model.predict(images, verbose=0)

    explanation = explainer.explain_instance(
        (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8),
        predict_fn,
        top_labels=top_labels,
        hide_color=0,
        num_samples=num_samples
    )

    # Get the explanation for top predicted class
    top_label = explanation.top_labels[0]
    image_explained, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=num_features,
        hide_rest=hide_rest
    )

    return explanation, mask, image_explained


def get_lime_feature_importance(explanation, label):
    """
    Extract feature importance scores from LIME explanation.

    Args:
        explanation: LIME explanation object
        label: Class label to get importances for

    Returns:
        Dict of {segment_id: importance_score}
    """
    local_exp = explanation.local_exp[label]
    importance = {seg_id: weight for seg_id, weight in local_exp}
    return importance


if __name__ == "__main__":
    print("[LIME] Module loaded.")
    print("[LIME] Use explain_with_lime(model, image) for pixel-region explanations.")
