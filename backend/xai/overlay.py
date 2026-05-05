"""
BT3068 — Heatmap Overlay Renderer
====================================
Overlays Grad-CAM and LIME heatmaps on original medical images.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Custom medical heatmap colormap: Blue → Green → Yellow → Red
MEDICAL_CMAP = LinearSegmentedColormap.from_list('medical', [
    (0.0, '#0000ff'),   # Cool blue (low activation)
    (0.25, '#00ff00'),  # Green
    (0.5, '#ffff00'),   # Yellow
    (0.75, '#ff8800'),  # Orange
    (1.0, '#ff0000'),   # Red (high activation / pathology)
])


def create_heatmap_overlay(original_image, heatmap, alpha=0.5, colormap=None):
    """
    Overlay a heatmap on the original medical image.

    Args:
        original_image: Original image (H, W, 3), values [0, 1] or [0, 255]
        heatmap: Activation heatmap (h, w), values [0, 1]
        alpha: Overlay opacity (0 = original only, 1 = heatmap only)
        colormap: Matplotlib colormap (default: medical)

    Returns:
        overlay: Blended image (H, W, 3), values [0, 255] as uint8
    """
    if colormap is None:
        colormap = MEDICAL_CMAP

    # Normalize original to [0, 255]
    if original_image.max() <= 1.0:
        original = (original_image * 255).astype(np.uint8)
    else:
        original = original_image.astype(np.uint8)

    # Resize heatmap to match original
    h, w = original.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply colormap
    heatmap_colored = (colormap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)

    # Blend
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay


def create_side_by_side(original_image, heatmap, alpha=0.5):
    """
    Create side-by-side comparison: Original | Heatmap Overlay.

    Returns:
        Combined image (H, W*2, 3)
    """
    overlay = create_heatmap_overlay(original_image, heatmap, alpha)

    if original_image.max() <= 1.0:
        original = (original_image * 255).astype(np.uint8)
    else:
        original = original_image.astype(np.uint8)

    combined = np.hstack([original, overlay])
    return combined


def find_activation_regions(heatmap, threshold=0.7, min_area=100):
    """
    Find high-activation bounding boxes in the heatmap.

    Args:
        heatmap: Activation heatmap (H, W), values [0, 1]
        threshold: Activation threshold for region detection
        min_area: Minimum region area in pixels

    Returns:
        List of bounding boxes: [(x, y, w, h, max_activation), ...]
    """
    # Threshold
    binary = (heatmap > threshold).astype(np.uint8) * 255

    # Resize if needed
    if binary.shape[0] < 50:
        scale = 380 // binary.shape[0]
        binary = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        heatmap_scaled = cv2.resize(heatmap, None, fx=scale, fy=scale)
    else:
        heatmap_scaled = heatmap

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            max_act = float(heatmap_scaled[y:y+h, x:x+w].max())
            regions.append({
                'x': int(x), 'y': int(y),
                'width': int(w), 'height': int(h),
                'max_activation': round(max_act, 3)
            })

    # Sort by activation (highest first)
    regions.sort(key=lambda r: r['max_activation'], reverse=True)
    return regions


def save_overlay(original, heatmap, save_path, alpha=0.5, dpi=150):
    """Save overlay visualization to file."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    if original.max() <= 1.0:
        orig_display = original
    else:
        orig_display = original / 255.0

    axes[0].imshow(orig_display)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    h, w = original.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    axes[1].imshow(heatmap_resized, cmap=MEDICAL_CMAP, vmin=0, vmax=1)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    overlay = create_heatmap_overlay(original, heatmap, alpha)
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay (α={alpha})', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"[OVERLAY] Saved to: {save_path}")
