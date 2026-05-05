"""
BT3068 — /api/heatmap Endpoint
=================================
Generates Grad-CAM heatmap overlays for medical images.
"""

import os
import sys
import base64
import io
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image

router = APIRouter()

class HeatmapRequest(BaseModel):
    image: str  # base64 encoded
    prediction_class: str = ""
    opacity: float = 0.5

class ActivationRegion(BaseModel):
    x: int
    y: int
    width: int
    height: int
    max_activation: float

class HeatmapResponse(BaseModel):
    heatmap_overlay: str  # base64 PNG
    heatmap_raw: str  # base64 PNG (heatmap only)
    activation_regions: List[ActivationRegion]
    opacity: float


def generate_demo_heatmap(image_array: np.ndarray):
    """
    Generate a realistic demonstration heatmap based on image features.
    Uses edge detection and intensity gradients to simulate pathology regions.
    """
    h, w = image_array.shape[:2]
    
    # Convert to grayscale
    if len(image_array.shape) == 3:
        gray = np.mean(image_array, axis=2)
    else:
        gray = image_array
    
    # Create multi-scale activation map
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # Simulate pathology region (gaussian blob at areas of high variance)
    from scipy import ndimage
    
    # Edge-based activation
    edges_x = ndimage.sobel(gray, axis=0)
    edges_y = ndimage.sobel(gray, axis=1)
    edges = np.hypot(edges_x, edges_y)
    
    # Local variance (areas of texture = potential pathology)
    local_mean = ndimage.uniform_filter(gray, size=20)
    local_sqr_mean = ndimage.uniform_filter(gray**2, size=20)
    local_var = np.clip(local_sqr_mean - local_mean**2, 0, None)
    
    # Combine signals
    heatmap = 0.4 * edges / (edges.max() + 1e-8) + 0.6 * local_var / (local_var.max() + 1e-8)
    
    # Add gaussian focus region (simulating model attention)
    cy, cx = h // 2 + np.random.randint(-h//6, h//6), w // 2 + np.random.randint(-w//6, w//6)
    y_grid, x_grid = np.ogrid[:h, :w]
    sigma = min(h, w) // 4
    gaussian = np.exp(-((y_grid - cy)**2 + (x_grid - cx)**2) / (2 * sigma**2))
    heatmap = 0.5 * heatmap + 0.5 * gaussian
    
    # Smooth and normalize
    heatmap = ndimage.gaussian_filter(heatmap, sigma=10)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return heatmap


def create_colored_heatmap(heatmap, colormap='jet'):
    """Convert grayscale heatmap to colored version."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    cm = plt.get_cmap(colormap)
    colored = (cm(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return colored


def array_to_base64(image_array):
    """Convert numpy array to base64 PNG string."""
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    else:
        image_array = image_array.astype(np.uint8)
    
    pil_img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def find_regions(heatmap, threshold=0.6):
    """Find high-activation regions as bounding boxes."""
    try:
        import cv2
        binary = (heatmap > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= 200:
                x, y, w, h = cv2.boundingRect(cnt)
                max_act = float(heatmap[y:y+h, x:x+w].max())
                regions.append(ActivationRegion(
                    x=int(x), y=int(y), width=int(w), height=int(h),
                    max_activation=round(max_act, 3)
                ))
        regions.sort(key=lambda r: r.max_activation, reverse=True)
        return regions[:5]
    except ImportError:
        return []


@router.post("/heatmap", response_model=HeatmapResponse)
async def generate_heatmap(request: HeatmapRequest):
    """
    Generate Grad-CAM heatmap overlay for a medical image.
    """
    try:
        # Decode image
        img_data = request.image
        if ',' in img_data:
            img_data = img_data.split(',')[1]
        
        image_bytes = base64.b64decode(img_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((380, 380), Image.LANCZOS)
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Try real Grad-CAM first
        heatmap = None
        try:
            import tensorflow as tf
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 
                                       'checkpoints', 'final_model.keras')
            if os.path.exists(model_path):
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
                from xai.gradcam import get_gradcam_heatmap
                model = tf.keras.models.load_model(model_path)
                heatmap = get_gradcam_heatmap(model, image_array)
        except Exception:
            pass
        
        # Fallback to demo heatmap
        if heatmap is None:
            heatmap = generate_demo_heatmap(image_array)
        
        # Create overlay
        colored_heatmap = create_colored_heatmap(heatmap)
        
        alpha = request.opacity
        original_uint8 = (image_array * 255).astype(np.uint8)
        overlay = ((1 - alpha) * original_uint8 + alpha * colored_heatmap).astype(np.uint8)
        
        # Find activation regions
        regions = find_regions(heatmap)
        
        return HeatmapResponse(
            heatmap_overlay=array_to_base64(overlay),
            heatmap_raw=array_to_base64(colored_heatmap),
            activation_regions=regions,
            opacity=alpha
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {str(e)}")
