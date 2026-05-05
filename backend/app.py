"""
BT3068 — FastAPI Backend Application
=======================================
Main application entry point for the Antigravity Neural Diagnostics system.
Serves both the API endpoints and the frontend static files.
"""

import os
import sys
import time
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.api.predict import router as predict_router
from backend.api.heatmap import router as heatmap_router
from backend.api.report import router as report_router

# ══════════════════════════════════════════
# Application Setup
# ══════════════════════════════════════════

app = FastAPI(
    title="BT3068 — Antigravity Neural Diagnostics",
    description="Multi-Disease Detection from Medical Images Using Transfer Learning",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(predict_router, prefix="/api", tags=["Prediction"])
app.include_router(heatmap_router, prefix="/api", tags=["Heatmap"])
app.include_router(report_router, prefix="/api", tags=["Report"])


# ══════════════════════════════════════════
# Metrics Endpoint
# ══════════════════════════════════════════

@app.get("/api/metrics")
async def get_metrics():
    """
    Return current model performance metrics.
    Based on achieved results: 95.67% accuracy on Chest X-Ray test set.
    """
    return {
        "accuracy": 0.9567,
        "auroc": 0.971,
        "f1_score": 0.957,
        "cohen_kappa": 0.809,
        "precision_weighted": 0.957,
        "recall_weighted": 0.957,
        "confusion_matrix": [[215, 19], [8, 382]],
        "per_class": {
            "NORMAL": {"precision": 0.964, "recall": 0.919, "f1": 0.941, "support": 234},
            "PNEUMONIA": {"precision": 0.953, "recall": 0.979, "f1": 0.966, "support": 390}
        },
        "model_info": {
            "backbone": "EfficientNetB4",
            "attention": "Squeeze-and-Excitation (SE)",
            "parameters": "19M",
            "input_size": "380x380",
            "training_phases": 2
        },
        "model_comparison": [
            {"name": "EfficientNetB4+SE (Ours)", "accuracy": 95.67},
            {"name": "EfficientNetB4+LRA", "accuracy": 94.04},
            {"name": "MRLA Ensemble", "accuracy": 96.0},
            {"name": "ResNet50", "accuracy": 91.2},
            {"name": "VGG16", "accuracy": 89.5},
            {"name": "DenseNet121", "accuracy": 93.1},
            {"name": "MobileNetV2", "accuracy": 88.7}
        ]
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "system": "ANTIGRAVITY Neural Diagnostics",
        "version": "1.0.0",
        "timestamp": time.time()
    }


# ══════════════════════════════════════════
# Static Files & Frontend
# ══════════════════════════════════════════

frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/js", StaticFiles(directory=os.path.join(frontend_dir, "js")), name="js")
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

    @app.get("/{filename:path}")
    async def serve_static_files(filename: str):
        file_path = os.path.join(frontend_dir, filename)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dir, "index.html"))


# Ensure uploads directory exists
os.makedirs(os.path.join(os.path.dirname(__file__), "uploads"), exist_ok=True)


if __name__ == "__main__":
    print("=" * 60)
    print("  ANTIGRAVITY NEURAL DIAGNOSTICS — BT3068")
    print("  Starting server on http://localhost:8000")
    print("=" * 60)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
