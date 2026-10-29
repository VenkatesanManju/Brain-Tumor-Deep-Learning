# NeuroVision AI: Explainable Deep Learning Framework for Multi-Domain Brain Tumor MRI Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg)](https://flask.palletsprojects.com/)
[![Validation](https://img.shields.io/badge/Validation%20Scans-4%2C664-green.svg)](#empirical-evaluation--clinical-benchmarks)
[![Ensemble Accuracy](https://img.shields.io/badge/Internal%20Acc-95.69%25-brightgreen.svg)](#empirical-evaluation--clinical-benchmarks)
[![External Generalization](https://img.shields.io/badge/External%20Acc-94.06%25-brightgreen.svg)](#empirical-evaluation--clinical-benchmarks)

---

## 📌 Executive Summary

Manual interpretation of multi-planar 2D brain Magnetic Resonance Imaging (MRI) is cognitive-heavy, time-consuming, and subject to inter-observer variability. While deep convolutional neural networks (CNNs) show promise, clinical translation is severely impeded by **four major failure modes**:

1. **Black-Box Opacity**: Standard neural networks output class probabilities without spatial or radiological rationale.
2. **Out-of-Distribution (OOD) Vulnerability**: Standard models produce high-confidence erroneous diagnoses when presented with non-brain images, artifacts, or photographic noise.
3. **Multi-Plane Volumetric Degradation**: Models trained largely on axial slices experience performance drops when tested against sagittal or coronal views.
4. **Cross-Hospital Domain Shift**: Discrepancies in scanner manufacturers, magnetic field strengths (1.5T vs. 3.0T), and sequence protocols degrade accuracy across different clinical centers.

**NeuroVision AI** resolves these challenges through an end-to-end framework featuring a **multi-architecture soft-voting ensemble** (ResNet50, EfficientNetB0, DenseNet121), an **automated MRI slice-view classifier** with orientation-conditioned calibration, a **multi-stage computer-vision OOD guardrail engine**, and an explainability suite driven by **Grad-CAM** and a novel **Ensemble Agreement Score (EAS)** metric.

---

## 🌟 Key Innovations & System Features

```
                                  [ Input MRI Slice (.png / .jpg / .mat) ]
                                                     │
                                                     ▼
                                     ┌───────────────────────────────┐
                                     │  Stage 1: CV OOD Guardrails   │
                                     │  - Color Variance Check       │
                                     │  - Circular Aperture Detector │
                                     │  - Spectral Entropy Bounds    │
                                     └───────────────┬───────────────┘
                                                     │ Passed
                                                     ▼
                                     ┌───────────────────────────────┐
                                     │  Stage 2: Slice View Detector │
                                     │  - Bilateral Symmetry (L-R)   │
                                     │  - Skull ROI Aspect Ratio     │
                                     │  → Axial / Coronal / Sagittal │
                                     └───────────────┬───────────────┘
                                                     │
                                                     ▼
                           ┌─────────────────────────┴─────────────────────────┐
                           │                                                   │
                           ▼                                                   ▼
            ┌─────────────────────────────┐                     ┌─────────────────────────────┐
            │   Multi-Model Prediction    │                     │   Explainable AI Suite      │
            │   - ResNet50 (Residual)     │                     │   - Grad-CAM Heatmaps       │
            │   - EfficientNetB0 (Scaled) │                     │   - Multi-Model Consensus   │
            │   - DenseNet121 (Dense)     │                     │   - Spatial IoU Overlap     │
            │   - Soft-Voting Ensemble    │                     │   - Feature Cosine Sim      │
            └──────────────┬──────────────┘                     └──────────────┬──────────────┘
                           │                                                   │
                           └─────────────────────────┬─────────────────────────┘
                                                     ▼
                                     ┌───────────────────────────────┐
                                     │ Stage 3: Decision Calibration │
                                     │ - Orientation-Prior Shift     │
                                     │ - EAS Agreement Scoring       │
                                     │ - Diagnostic Confidence Level │
                                     └───────────────┬───────────────┘
                                                     │
                                                     ▼
                                     ┌───────────────────────────────┐
                                     │   Clinical Delivery Layer     │
                                     │   - Interactive Web Console   │
                                     │   - Real-time Saliency Slider │
                                     │   - Automated PDF Report      │
                                     └───────────────────────────────┘
```

- **Multi-Architecture Soft-Voting Ensemble**: Combines ResNet50 (macro structural anatomy), EfficientNetB0 (micro-texture lesions), and DenseNet121 (gradient propagation across contrast boundaries) to eliminate single-architecture bias.
- **Slice-View Orientation Calibration**: Automatically computes Bilateral Reflection Symmetry and ROI Skull Aspect Ratio to detect *Axial*, *Coronal*, or *Sagittal* planes, applying orientation-conditioned calibration to overcome multi-plane drops.
- **Pre-Inference OOD Guardrails**: Evaluates grayscale variance, color saturation anomalies, circular aperture contours (e.g., ocular fundus images), and spectral frequency distributions to reject non-brain uploads before inference.
- **Ensemble Agreement Score (EAS)**: Quantifies inter-model visual and latent consensus by coupling Grad-CAM spatial Intersection over Union (IoU) with penultimate-layer Cosine Similarity:
  $$\text{EAS} = \alpha \cdot \text{IoU}(\mathcal{M}_A, \mathcal{M}_B, \mathcal{M}_C) + (1 - \alpha) \cdot \text{CosineSim}(f_A, f_B, f_C)$$
- **Full-Scale Empirical Validation ($N = 4,664$ scans)**: Validated on 1,600 held-out primary scans and 3,064 cross-hospital external scans, supported by non-parametric Mann-Whitney $U$ hypothesis testing ($p < 10^{-28}$).
- **Interactive Web App & Clinical PDF Engine**: Flask-powered medical console with interactive Grad-CAM heatmap blending, confidence bar charts, and one-click downloadable clinical reports.

---

## 📊 Empirical Evaluation & Clinical Benchmarks

The system was evaluated against two independent clinical cohorts:
- **Internal Cohort ($N = 1,600$)**: Multi-class Kaggle Brain Tumor MRI Dataset (Healthy, Glioma, Meningioma, Pituitary).
- **External Multi-Hospital Cohort ($N = 3,064$)**: Cheng / Figshare Multi-Center Clinical Dataset for cross-domain generalization testing.

### Master Classification Performance

| Model Architecture | Internal Accuracy | External Accuracy | Generalization Gap ($\Delta$) | Internal Macro F1 | External Macro F1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **ResNet50** | 95.25% | 90.70% | 4.55% | 0.9516 | 0.9269 |
| **EfficientNetB0** | 95.38% | 92.43% | 2.95% | 0.9528 | 0.9405 |
| **DenseNet121** | 95.44% | 90.54% | 4.90% | 0.9534 | 0.9130 |
| **NeuroVision Ensemble (Ours)** | **95.69%** | **94.06%** | **1.63%** | **0.9559** | **0.9513** |

> **Key Finding**: The soft-voting ensemble reduces the cross-hospital generalization gap from up to **4.90% down to only 1.63%**, while lifting external test accuracy to **94.06%** (+1.63% over the best individual model).

### Statistical Significance of Explainability Metric (EAS)

Mann-Whitney $U$ tests performed on per-sample Ensemble Agreement Scores prove that correctly classified scans display significantly higher consensus than misclassifications:

| Evaluation Set | Correct Mean EAS | Error Mean EAS | Mann-Whitney $U$ | $p$-value | Cliff's $\delta$ | Cohen's $d$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Internal Test ($N = 1,600$)** | **0.924** | 0.531 | 95,173.0 | $1.62 \times 10^{-29}$ | 0.8019 (Large) | 3.9673 |
| **External Test ($N = 3,064$)** | **0.887** | 0.589 | 470,756.0 | $7.92 \times 10^{-74}$ | 0.8043 (Large) | 2.4843 |

---

## 📁 Repository Structure

```
.
├── app.py                                   # Clinical Flask Web Application & PDF Service
├── main.py                                  # End-to-end training & evaluation pipeline CLI
├── run_pipeline.sh                          # One-click execution shell script
├── requirements.txt                         # Python dependencies specification
├── src/                                     # Core modular packages
│   ├── config.py                            # Project paths, label maps, hyperparameters
│   ├── data.py                              # PyTorch Dataset loaders & augmentations
│   ├── train.py                             # Training loops, LR scheduling, checkpointing
│   ├── ensemble.py                          # Soft-voting fusion engine
│   ├── gradcam.py                           # Multi-model Grad-CAM generator
│   ├── eas.py                               # Ensemble Agreement Score calculation
│   ├── view_classifier.py                   # Bilateral symmetry & slice-view calibration
│   ├── dedup_check.py                       # Data leakage & perceptual hashing check
│   ├── convert_external.py                  # .mat Figshare to PNG converter
│   ├── evaluate_internal.py                 # Internal test set benchmark
│   ├── evaluate_external.py                 # External cross-center benchmark
│   ├── failure_analysis.py                  # Error categorization and outlier audit
│   └── utils.py                             # Image normalization & MAT loading helpers
├── templates/                               # Web UI HTML templates
│   └── index.html                           # Modern glassmorphism Tailwind UI
├── tests/                                   # Automated test suite
│   └── test_dataset_class_mapping.py
├── outputs/                                 # Benchmark outputs & reports
│   ├── checkpoints/                         # Saved model weights (ResNet50, DenseNet, EfficientNet)
│   ├── report_figures/                      # Publication figures (ROC, PR, confusion matrices)
│   └── final_metrics.json                   # Evaluation metrics JSON
├── .gitignore                               # Git ignore rules for virtualenvs, data & weights
├── .gitattributes                           # Platform line-ending & Git LFS configurations
├── .env.example                             # Environment variables template
└── README.md                                # Comprehensive project documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites

- **Python**: 3.10 or higher recommended.
- **Hardware**: CUDA-compatible GPU (e.g., NVIDIA RTX 3050/3060/4060 or better) recommended; CPU execution is fully supported.
- **Git**

### 2. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 3. Create a Virtual Environment & Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

> **Note on PyTorch (CUDA GPU Acceleration)**:  
> If using an NVIDIA GPU with CUDA 12.1, install PyTorch with CUDA support:
> ```bash
> pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
> ```

---

## 🗄️ Dataset Organization

The framework expects datasets inside the `data/` directory:

```
data/
├── primary/
│   ├── notumor/
│   ├── glioma/
│   ├── meningioma/
│   └── pituitary/
├── external/
│   ├── glioma/
│   ├── meningioma/
│   └── pituitary/
└── external_raw/            # Optional raw .mat files from Cheng/Figshare
```

### Dataset Sources:
1. **Primary Dataset**: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparamb/brain-tumor-mri-dataset) (4 classes: No Tumor, Glioma, Meningioma, Pituitary).
2. **External Dataset**: [Figshare Brain Tumor Dataset (Cheng et al.)](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) (3 tumor classes across 3,064 T1-weighted contrast-enhanced slices).

To convert raw Figshare `.mat` files to PNG:
```bash
python src/convert_external.py --input-root data/external_raw --output-root data/external
```

---

## 💻 How to Run

### 1. Launch the Interactive Web Application

Start the clinical decision support interface:

```bash
python app.py
```

The server automatically detects an open port (default: `5050`) and launches your default browser at:
```
http://127.0.0.1:5050/
```

#### Application Features:
- **Instant Upload**: Drag & drop standard `.png`, `.jpg`, or `.mat` MRI scans.
- **OOD Guardrail Alert**: Rejects non-brain or corrupted inputs with clear feedback.
- **Slice-View Plane Badge**: Displays detected orientation (*Axial*, *Coronal*, *Sagittal*) and confidence.
- **Multi-Model Probability Distribution**: Compare individual predictions side by side with the soft-voting ensemble.
- **Interactive Grad-CAM Visualizer**: Toggle between individual model heatmaps or the combined consensus map, with dynamic opacity adjustment.
- **One-Click PDF Report**: Generates a downloadable clinical report with diagnostic metrics, findings narrative, and heatmaps.

---

### 2. Run the Full Training & Evaluation Pipeline

To train the models from scratch and run internal/external evaluations:

```bash
# Using the Python entry point:
python main.py --epochs 20 --batch-size 32 --lr 1e-4

# Or using the shell script:
bash run_pipeline.sh
```

To skip data deduplication and jump straight to training/eval:
```bash
python main.py --skip-early-stages --epochs 10
```

---

### 3. Run Automated Unit Tests

```bash
pytest tests/
```

---

## 📦 Model Weights & Checkpoints

Due to GitHub's file size limits (100MB per file hard cap, 50MB warning threshold), pre-trained model weights are tracked via `.gitignore`:
- `ResNet50.pt` (~90 MB)
- `DenseNet121.pt` (~28 MB)
- `EfficientNetB0.pt` (~16 MB)

### Recommended Options for Sharing Model Weights:
1. **GitHub Releases** *(Recommended)*: Download checkpoints attached to the latest Release into `outputs/checkpoints/`.
2. **Git LFS (Large File Storage)**: If you prefer tracking weights within Git:
   ```bash
   git lfs install
   git lfs track "*.pt"
   ```
3. **Hugging Face Hub**: Upload weights to Hugging Face Model Hub and download on demand.
