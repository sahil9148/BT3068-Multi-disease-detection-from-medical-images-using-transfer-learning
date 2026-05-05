# 🧠 BT3068 — Multi-Disease Detection from Medical Images Using Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-Academic-purple.svg)](#)

> **ANTIGRAVITY Neural Diagnostics** — A unified AI system for multi-disease detection across Brain, Lung, Eye, and Skin domains using EfficientNetB4 + Squeeze-and-Excitation Channel Attention + Ensemble Methods.

---

## 🎯 Project Overview

This project addresses the global shortage of expert radiologists by building a **unified Multi-Disease Detection system** capable of simultaneously diagnosing conditions across four major disease domains from medical images.

### Disease Domains

| Domain | Diseases | Imaging Modality |
|--------|----------|-----------------|
| 🧠 Brain | Alzheimer's, Brain Tumour, Parkinson's | MRI |
| 🫁 Lung | COVID-19, Pneumonia, Lung Cancer | CT, X-Ray |
| 👁 Eye | Glaucoma, Diabetic Retinopathy, AMD | OCT, Fundus |
| 🧬 Skin | Melanoma, Carcinoma, Naevus | Dermoscopy |

### Current Results

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **95.67%** |
| Precision (Weighted) | 95.7% |
| Recall (Weighted) | 95.7% |
| F1-Score (Weighted) | 95.7% |
| False Negative Rate (Pneumonia) | 2.1% |

---

## 🏗️ Architecture

### ML Pipeline
- **Backbone:** EfficientNetB4 (ImageNet pre-trained)
- **Attention:** Squeeze-and-Excitation (SE) Channel Attention Block
- **Training:** Two-Phase Transfer Learning (Frozen → Fine-tune top-100 layers)
- **Ensemble:** MobileNetV2 + DenseNet121 + ResNet101 with Stacking Meta-Learner
- **Explainability:** Grad-CAM + LIME heatmap overlays

### Tech Stack
- **Deep Learning:** TensorFlow 2.x + Keras, PyTorch 2.0
- **Backend:** FastAPI (Python)
- **Frontend:** Antigravity UI (HTML/CSS/JS)
- **Visualization:** Chart.js, Matplotlib, Seaborn

---

## 📁 Project Structure

```
BT3068-MultiDiseaseDetection/
├── data/preprocessing/        # Data augmentation, class weights, segmentation
├── models/                    # Model architectures and training scripts
├── evaluation/                # Metrics, confusion matrix, ROC curves
├── xai/                       # Grad-CAM, LIME explainability
├── webapp/                    # FastAPI backend + Antigravity frontend
│   ├── app.py                 # FastAPI application
│   ├── api/                   # API endpoints
│   └── frontend/              # Antigravity UI
├── notebooks/                 # Jupyter notebooks for EDA and training
└── literature/                # Research references
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
Place your medical image datasets in the `data/` directory:
- `data/chest_xray/` — Kaggle Chest X-Ray (train/test/val splits)
- `data/brain_mri/` — Alzheimer's MRI dataset
- `data/retinal_fundus/` — APTOS 2019
- `data/skin/` — HAM10000 / ISIC

### 3. Train Model
```bash
# Phase 1: Frozen base training
python models/train_phase1.py --dataset chest_xray --epochs 20

# Phase 2: Fine-tuning
python models/train_phase2.py --dataset chest_xray --epochs 30
```

### 4. Run Web Application
```bash
# Start FastAPI backend
cd webapp
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Open frontend
# Navigate to http://localhost:8000 in your browser
```

---

## 📊 Model Performance Comparison

| Model | Accuracy | Source |
|-------|----------|--------|
| **Ours: EfficientNetB4 + SE Block** | **95.67%** | This project |
| Natarajan et al. — EfficientNetB4+LRA | 94.04% | Paper P1 |
| MRLA Ensemble | ~96% | Paper P11 |
| AlexNet+SVM (Alzheimer's) | 99.75% | Paper P19 |

---

## 👥 Team

| Name | Roll Number | Role |
|------|-------------|------|
| Vipin Tiwari | CS-23411432 | Team Member |
| Yash Grover | CS-23411108 | Team Member |
| Varun Sharma | CS-2341097 | Team Member |
| Sahil Arora | CS-2341220 | Team Member |
| Yash Mahato | CS-2341851 | Team Member |

**Guide:** Mr. Anand Kumar  
**Institution:** IILM University, Greater Noida, India  
**Department:** School of Computer Science and Engineering

---

## 📚 Key References

1. Natarajan et al. (2024). Multi-disease detection using EfficientNetB4+LRA. *Neural Computing and Applications*, Springer.
2. Hu, Shen & Sun (2018). Squeeze-and-Excitation Networks. *CVPR*.
3. Tan & Le (2019). EfficientNet: Rethinking model scaling. *ICML*.
4. Ayana et al. (2024). Multistage Transfer Learning review. *AI Review*, 57:232.

---

## 📄 License

This project is developed for academic purposes as part of the B.Tech degree program at IILM University.
