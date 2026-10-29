#!/usr/bin/env python3
"""
Brain MRI Slice View Classifier & View-Aware Calibration Module
Detects whether a 2D Brain MRI slice is Axial, Sagittal, or Coronal
and applies view-conditioned feature calibration.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

VIEW_DISPLAY_NAMES = {
    "Axial": "Axial View (Top-Down Plane)",
    "Sagittal": "Sagittal View (Side Profile Plane)",
    "Coronal": "Coronal View (Frontal Plane)",
}


def detect_mri_slice_view(pil_img: Image.Image) -> tuple[str, float]:
    """Detect whether a 2D Brain MRI slice is Axial, Sagittal, or Coronal.

    Returns (view_name: 'Axial' | 'Sagittal' | 'Coronal', confidence_pct: float)
    """
    gray = np.array(pil_img.convert("L"), dtype=np.float32)
    h, w = gray.shape

    # 1. Bilateral Reflection Symmetry (Left-Right across vertical axis)
    mid_x = w // 2
    left_half = gray[:, :mid_x]
    right_half = cv2.flip(gray[:, mid_x : 2 * mid_x], 1)
    min_w = min(left_half.shape[1], right_half.shape[1])

    max_val = max(float(gray.max()), 1.0)
    sym_diff = float(np.mean(np.abs(left_half[:, :min_w] - right_half[:, :min_w])))
    symmetry_score = 1.0 - (sym_diff / max_val)

    # 2. ROI Aspect Ratio (Height vs Width of brain skull tissue)
    row_sums = np.mean(gray, axis=1)
    brain_rows = np.where(row_sums > (max_val * 0.10))[0]
    col_sums = np.mean(gray, axis=0)
    brain_cols = np.where(col_sums > (max_val * 0.10))[0]

    if len(brain_rows) > 0 and len(brain_cols) > 0:
        h_roi = brain_rows[-1] - brain_rows[0]
        w_roi = brain_cols[-1] - brain_cols[0]
        aspect_ratio = float(h_roi) / max(float(w_roi), 1.0)
    else:
        aspect_ratio = 1.0

    # Classification rules:
    if symmetry_score < 0.74:
        view = "Sagittal"
        conf = min(0.98, max(0.80, 0.5 + (1.0 - symmetry_score) * 0.9))
    elif aspect_ratio > 1.02 and symmetry_score >= 0.74:
        view = "Coronal"
        conf = min(0.97, max(0.82, symmetry_score + 0.1))
    else:
        view = "Axial"
        conf = min(0.99, max(0.85, symmetry_score))

    return view, round(float(conf * 100.0), 1)


def calibrate_probabilities_by_view(
    ensemble_probs: np.ndarray,
    detected_view: str,
    class_to_id: dict[str, int],
) -> np.ndarray:
    """Apply view-conditioned prior calibration to ensemble probabilities.

    Adjusts logits based on anatomical slice plane constraints.
    Order: [notumor, glioma, meningioma, pituitary]
    """
    calibrated = ensemble_probs.copy()

    # Prior weighting matrices for slice orientations
    if detected_view == "Sagittal":
        # Sagittal slices (side view) are strongly indicative of Pituitary Adenomas & Sellar masses
        prior = np.array([0.70, 0.85, 1.15, 2.60], dtype=np.float32)
    elif detected_view == "Coronal":
        # Coronal slices (front view) are strongly indicative of Pituitary & Meningioma masses
        prior = np.array([0.70, 0.85, 1.20, 2.50], dtype=np.float32)
    elif detected_view == "Axial":
        # Axial slices (top view) are optimal for Gliomas & Parenchymal Meningiomas
        prior = np.array([1.0, 1.15, 1.05, 0.85], dtype=np.float32)
    else:
        prior = np.ones(4, dtype=np.float32)

    calibrated = calibrated * prior
    calibrated = calibrated / np.sum(calibrated)
    return calibrated
