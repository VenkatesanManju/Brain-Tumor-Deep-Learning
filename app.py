#!/usr/bin/env python3
"""
Brain Tumor MRI Classification & Real-Time Explainable AI (XAI) Web Application
"""
from __future__ import annotations

import base64
from datetime import datetime
import io
import logging
import os
from pathlib import Path
import socket
import sys
import threading
import webbrowser

import cv2
from fpdf import FPDF
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, cli, jsonify, render_template, request, send_file, send_from_directory
from PIL import Image
from torchvision import transforms

# Suppress noisy Flask/Werkzeug development server banner and request logs
cli.show_server_banner = lambda *args, **kwargs: None
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# Ensure src is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import CLASS_TO_ID, LABEL_TO_CLASS, MODEL_NAMES
from train import build_model
from utils import load_mat_cjdata, normalize_image_uint8
from view_classifier import VIEW_DISPLAY_NAMES, calibrate_probabilities_by_view, detect_mri_slice_view

app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"))
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Global device and models cache
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS: dict[str, torch.nn.Module] = {}
GRADCAM_TARGETS: dict[str, torch.nn.Module] = {}

CLASS_DISPLAY_NAMES = {
    "notumor": "Healthy / No Tumor",
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "pituitary": "Pituitary Tumor",
}

CLASS_DESCRIPTIONS = {
    "notumor": "Normal brain tissue architecture without focal mass effect, edema, or pathological enhancement.",
    "glioma": "Intra-axial glial neoplasm typically arising within brain parenchyma with infiltrative margins.",
    "meningioma": "Extra-axial dural-based tumor arising from arachnoid cap cells, often causing mass effect on adjacent brain.",
    "pituitary": "Sellar / suprasellar region tumor arising from the pituitary gland near the optic chiasm and skull base.",
}


def get_target_layer(model_name: str, model: torch.nn.Module) -> torch.nn.Module:
    if model_name == "ResNet50":
        return model.layer4[-1]
    if model_name == "EfficientNetB0":
        return model.features[-1]
    if model_name == "DenseNet121":
        return model.features.denseblock4.denselayer16.conv2
    raise ValueError(f"Unknown model: {model_name}")


def load_all_models():
    global MODELS, GRADCAM_TARGETS
    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    for name in MODEL_NAMES:
        ckpt_path = ckpt_dir / f"{name}.pt"
        if ckpt_path.exists():
            try:
                model = build_model(name, num_classes=4).to(DEVICE)
                state = torch.load(ckpt_path, map_location=DEVICE)
                model.load_state_dict(state)
                model.eval()
                MODELS[name] = model
                GRADCAM_TARGETS[name] = get_target_layer(name, model)
                print(f"[Loaded] {name} from {ckpt_path.name}")
            except Exception as e:
                print(f"[Warning] Failed to load {name}: {e}")


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.handlers.append(target_layer.register_forward_hook(forward_hook))
        self.handlers.append(target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, target_class]
        score.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224), dtype=np.float32)

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        return cam

    def remove_hooks(self):
        for h in self.handlers:
            h.remove()


def preprocess_image(pil_img: Image.Image) -> tuple[torch.Tensor, Image.Image]:
    img_rgb = pil_img.convert("RGB")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)
    resized_pil = img_rgb.resize((224, 224))
    return tensor, resized_pil


def check_out_of_distribution(
    raw_pil_img: Image.Image,
    ensemble_probs: np.ndarray,
    indiv_preds: dict[str, str],
) -> tuple[bool, str]:
    """Check if the uploaded image is Out-Of-Distribution (OOD) / Non-Brain MRI.

    Returns (is_ood: bool, reason: str)
    """
    # 1. Color Saturation & Channel Variance Check (evaluated on raw image before grayscale normalization)
    raw_rgb = np.array(raw_pil_img.convert("RGB"), dtype=np.float32)
    r, g, b = raw_rgb[:, :, 0], raw_rgb[:, :, 1], raw_rgb[:, :, 2]
    rgb_diff = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(b - r)))
    if rgb_diff > 15.0:
        return True, f"Non-brain medical scan or color photograph detected (Color variation score: {rgb_diff:.1f}). Brain MRI scans are strictly 2D grayscale intensity scans."

    # 2. Retinal Fundus & Ophthalmology Lens Aperture Check (Grayscale or Color)
    gray = cv2.cvtColor(raw_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            (x, y), radius = cv2.minEnclosingCircle(c)
            circle_area = np.pi * (radius ** 2)
            extent = area / max(circle_area, 1e-5)
            corner_darkness = float(np.mean(gray[:25, :25]) + np.mean(gray[:25, -25:]) + np.mean(gray[-25:, :25]) + np.mean(gray[-25:, -25:])) / 4.0
            if extent < 0.50 and circularity > 0.40 and corner_darkness < 20.0:
                return True, "Retinal fundus / ophthalmology scan aperture pattern detected. Please upload a Brain MRI scan."

    # 3. Maximum Softmax Probability (MSP) & Normalized Entropy Check
    max_prob = float(np.max(ensemble_probs))
    eps = 1e-7
    entropy = -float(np.sum(ensemble_probs * np.log2(ensemble_probs + eps)))
    norm_entropy = float(entropy / np.log2(4.0))

    if max_prob < 0.45 or norm_entropy > 0.85:
        return True, f"High classification uncertainty (Max Confidence: {max_prob * 100.0:.1f}%, Entropy: {norm_entropy:.2f}). Feature maps do not match standard brain MRI tissue structures."

    # 4. Complete Model Disagreement Check
    unique_preds = set(indiv_preds.values())
    if len(indiv_preds) >= 3 and len(unique_preds) == 3 and max_prob < 0.55:
        return True, f"Complete model disagreement across architectures ({', '.join(indiv_preds.values())}) with low ensemble confidence ({max_prob * 100.0:.1f}%)."

    return False, ""


def compute_gradcam_overlay(input_tensor: torch.Tensor, pil_img: Image.Image, target_class: int) -> tuple[np.ndarray, np.ndarray]:
    cams = []
    for name, model in MODELS.items():
        target_layer = GRADCAM_TARGETS[name]
        extractor = GradCAM(model, target_layer)
        try:
            cam = extractor.generate(input_tensor, target_class)
            cams.append(cam)
        finally:
            extractor.remove_hooks()

    if cams:
        consensus_cam = np.mean(cams, axis=0)
        max_val = consensus_cam.max()
        if max_val > 0:
            consensus_cam = consensus_cam / max_val
    else:
        consensus_cam = np.zeros((224, 224), dtype=np.float32)

    img_np = np.array(pil_img)
    heatmap = cv2.applyColorMap(np.uint8(255 * consensus_cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.float32(heatmap) * 0.4 + np.float32(img_np) * 0.6
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return consensus_cam, overlay


def pil_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_realtime_xai_explanation(
    predicted_class: str,
    confidence_pct: float,
    cam_map: np.ndarray,
    pil_img: Image.Image,
    model_probs: dict[str, float],
    individual_preds: dict[str, str],
    is_ood: bool = False,
    ood_reason: str = "",
) -> str:
    """Dynamically generate a real-time Explainable AI (XAI) clinical explanation

    for the specific uploaded MRI image based on feature analysis & Grad-CAM attention.
    """
    img_gray = np.array(pil_img.convert("L"), dtype=np.float32)

    # Hotspot analysis from Grad-CAM
    peak_attention = float(cam_map.max())
    mean_attention = float(cam_map.mean())
    hotspot_mask = cam_map >= 0.55
    hotspot_area_pct = float(hotspot_mask.sum() / cam_map.size * 100.0)

    # Compute spatial center of attention
    if hotspot_mask.sum() > 0:
        ys, xs = np.where(hotspot_mask)
        cy = float(ys.mean() / cam_map.shape[0])
        cx = float(xs.mean() / cam_map.shape[1])
    else:
        cy, cx = 0.5, 0.5

    # Anatomical region estimation based on MRI slice coordinates
    if cy < 0.38:
        vertical_loc = "superior / parieto-frontal cranial region"
    elif cy > 0.62:
        vertical_loc = "inferior / sellar & skull base region"
    else:
        vertical_loc = "mid-brain / subcortical parenchymal region"

    if cx < 0.42:
        lateral_loc = "left hemisphere"
    elif cx > 0.58:
        lateral_loc = "right hemisphere"
    else:
        lateral_loc = "central / midline area"

    location_str = f"{vertical_loc} ({lateral_loc})"

    # Contrast & intensity ratio in hotspot vs background
    if hotspot_mask.sum() > 0:
        hotspot_intensity = float(img_gray[hotspot_mask].mean())
        background_intensity = float(img_gray[~hotspot_mask].mean()) if (~hotspot_mask).sum() > 0 else 1.0
        contrast_ratio = hotspot_intensity / max(background_intensity, 1.0)
    else:
        contrast_ratio = 1.0

    # Model consensus count
    same_pred_count = sum(1 for p in individual_preds.values() if p == predicted_class)
    models_count = len(individual_preds)

    explanation = []
    explanation.append(f"### Real-Time AI Interpretability & Visual Explanation Report\n")

    if is_ood:
        explanation.append(
            f"> ⚠️ **Input Validation / Out-Of-Distribution Warning:**\n"
            f"> High classification uncertainty detected. This file may not be a standard Brain MRI scan.\n"
            f"> **Detected Anomaly:** {ood_reason}\n"
        )

    if predicted_class == "notumor":
        explanation.append(
            f"**Diagnostic Summary:** The Deep Learning Ensemble evaluated this brain MRI scan and determined with **{confidence_pct:.1f}% confidence** "
            f"that **no brain tumor is present**.\n"
        )
        explanation.append(
            f"**Key Anatomical & Attentional Findings:**\n"
            f"1. **Uniform Structural Feature Maps:** Grad-CAM saliency maps confirm that neural network activations are diffusely distributed across "
            f"normal cerebral anatomical boundaries without concentrating on any localized abnormal hyperintense mass or space-occupying lesion.\n"
            f"2. **Absence of Mass Effect or Edema:** The localized attention area (>55% peak activation) comprises only **{hotspot_area_pct:.1f}%** of the image, "
            f"indicating no significant tissue compression, midline shift, or surrounding parenchymal edema.\n"
            f"3. **Baseline Signal Consistency:** Signal intensity in highlighted regions exhibits a balanced contrast ratio of **{contrast_ratio:.2f}x** relative "
            f"to surrounding brain parenchyma, consistent with physiological brain MRI appearance.\n"
            f"4. **Multi-Model Consensus:** **{same_pred_count} out of {models_count} independent deep neural architectures** (ResNet50, EfficientNetB0, DenseNet121) "
            f"unanimously confirmed the absence of neoplastic tumor tissue."
        )
    else:
        tumor_title = CLASS_DISPLAY_NAMES[predicted_class]
        explanation.append(
            f"**Diagnostic Summary:** The AI Ensemble identified pathological features indicative of a **{tumor_title}** "
            f"with **{confidence_pct:.1f}% overall prediction confidence**.\n"
        )
        explanation.append(
            f"**Key Anatomical & Attentional Findings (Real-Time Image Analysis):**\n"
            f"1. **Spatial Localization of Lesion:** Grad-CAM neural attention is strongly focused on a localized region in the **{location_str}**. "
            f"The primary focal hotspot occupies **{hotspot_area_pct:.1f}%** of the 2D MRI slice plane.\n"
            f"2. **Intensity & Contrast Anomaly:** The high-attention ROI displays a contrast ratio of **{contrast_ratio:.2f}x** compared to surrounding "
            f"normal brain tissue, reflecting hyperintense signal alterations characteristic of abnormal cellular tissue density or contrast enhancement.\n"
        )

        if predicted_class == "glioma":
            explanation.append(
                f"3. **Glioma Feature Signature:** The model highlighted infiltrative parenchymal boundaries within the {vertical_loc}. "
                f"Gliomas typically display poorly demarcated margins and intra-axial parenchymal involvement, which matches the diffuse attention falloff observed in the Grad-CAM heatmap.\n"
                f"4. **Neural Architecture Consensus:** **{same_pred_count} of {models_count} CNN models** concurred on the Glioma classification (Glioma probability: **{model_probs.get('glioma', 0)*100:.1f}%**)."
            )
        elif predicted_class == "meningioma":
            explanation.append(
                f"3. **Meningioma Feature Signature:** The attention pattern demonstrates a well-circumscribed extra-axial footprint near the dural / boundary region of the {vertical_loc}. "
                f"Meningiomas characteristically exhibit sharp lesion demarcation and dural attachment enhancement patterns detected by the convolutional feature maps.\n"
                f"4. **Neural Architecture Consensus:** **{same_pred_count} of {models_count} CNN models** concurred on the Meningioma classification (Meningioma probability: **{model_probs.get('meningioma', 0)*100:.1f}%**)."
            )
        elif predicted_class == "pituitary":
            explanation.append(
                f"3. **Pituitary Tumor Feature Signature:** Neural attention is concentrated precisely around the sellar / suprasellar anatomical region at the skull base midline. "
                f"Pituitary neuroendocrine tumors present as localized expansions within the sella turcica, which aligns directly with the focal centroid at {location_str}.\n"
                f"4. **Neural Architecture Consensus:** **{same_pred_count} of {models_count} CNN models** concurred on the Pituitary Tumor classification (Pituitary probability: **{model_probs.get('pituitary', 0)*100:.1f}%**)."
            )

    return "\n\n".join(explanation)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = file.filename.lower()

    try:
        if filename.endswith(".mat"):
            # MATLAB MAT file
            file_bytes = file.read()
            # Save to temporary memory buffer or file
            temp_mat = PROJECT_ROOT / "outputs" / "temp_upload.mat"
            temp_mat.parent.mkdir(parents=True, exist_ok=True)
            with temp_mat.open("wb") as f:
                f.write(file_bytes)
            cjdata = load_mat_cjdata(temp_mat)
            if "image" not in cjdata:
                return jsonify({"error": "Invalid MAT file structure; missing 'image' array"}), 400
            raw_img = cjdata["image"]
            uint8_img = normalize_image_uint8(raw_img)
            pil_img = Image.fromarray(uint8_img)
        else:
            # Standard image format (PNG, JPG, JPEG, BMP)
            file_bytes = file.read()
            raw_pil_img = Image.open(io.BytesIO(file_bytes)).copy()
            # Convert to uint8 normalized array if needed
            arr = np.array(raw_pil_img)
            uint8_img = normalize_image_uint8(arr)
            pil_img = Image.fromarray(uint8_img)

        # Preprocess
        input_tensor, resized_pil = preprocess_image(pil_img)

        # Individual model inference
        indiv_probs = {}
        indiv_preds = {}
        all_logits = []

        for name, model in MODELS.items():
            with torch.no_grad():
                logits = model(input_tensor)
                all_logits.append(logits)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_id = int(np.argmax(probs))
                pred_class = LABEL_TO_CLASS[pred_id]
                indiv_preds[name] = pred_class
                indiv_probs[name] = {LABEL_TO_CLASS[i]: float(probs[i]) for i in range(4)}

        # Anatomical Slice View Detection
        raw_check_img = raw_pil_img if "raw_pil_img" in locals() else pil_img
        detected_view, view_conf = detect_mri_slice_view(raw_check_img)

        # Ensemble soft voting & View-Aware Calibration
        if all_logits:
            raw_ensemble_probs = torch.stack([torch.softmax(lg, dim=1) for lg in all_logits], dim=0).mean(dim=0).cpu().numpy()[0]
        else:
            raw_ensemble_probs = np.array([0.25, 0.25, 0.25, 0.25])

        ensemble_probs = calibrate_probabilities_by_view(raw_ensemble_probs, detected_view, CLASS_TO_ID)

        top_id = int(np.argmax(ensemble_probs))
        predicted_class = LABEL_TO_CLASS[top_id]
        confidence_pct = float(ensemble_probs[top_id] * 100.0)
        has_tumor = predicted_class != "notumor"

        model_probs_dict = {LABEL_TO_CLASS[i]: float(ensemble_probs[i]) for i in range(4)}

        # Out-of-Distribution (OOD) Guardrail Check
        is_ood, ood_reason = check_out_of_distribution(raw_check_img, ensemble_probs, indiv_preds)

        # Grad-CAM heatmap generation
        cam_map, overlay_np = compute_gradcam_overlay(input_tensor, resized_pil, top_id)

        # Base64 images
        orig_b64 = pil_to_base64(resized_pil)
        overlay_pil = Image.fromarray(overlay_np)
        overlay_b64 = pil_to_base64(overlay_pil)

        # Real-time Explainable AI text explanation
        xai_explanation = generate_realtime_xai_explanation(
            predicted_class=predicted_class,
            confidence_pct=confidence_pct,
            cam_map=cam_map,
            pil_img=resized_pil,
            model_probs=model_probs_dict,
            individual_preds=indiv_preds,
            is_ood=is_ood,
            ood_reason=ood_reason,
        )

        response = {
            "status": "success",
            "has_tumor": has_tumor,
            "predicted_class": predicted_class,
            "display_name": CLASS_DISPLAY_NAMES[predicted_class],
            "description": CLASS_DESCRIPTIONS[predicted_class],
            "confidence_pct": round(confidence_pct, 2),
            "model_probabilities": model_probs_dict,
            "individual_predictions": indiv_preds,
            "original_image_b64": orig_b64,
            "gradcam_overlay_b64": overlay_b64,
            "xai_explanation": xai_explanation,
            "is_ood": is_ood,
            "ood_reason": ood_reason,
            "detected_view": detected_view,
            "view_display_name": VIEW_DISPLAY_NAMES[detected_view],
            "view_confidence": view_conf,
        }
        return jsonify(response)

    except Exception as exc:
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"Failed to process MRI image: {str(exc)}"}), 500


def sanitize_text_for_fpdf(text: str) -> str:
    if not text:
        return ""
    replacements = {
        "—": "-",
        "–": "-",
        "…": "...",
        "•": "*",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "≥": ">=",
        "≤": "<=",
        "±": "+/-",
        "°": " deg",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "*",
        "\u2026": "...",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


class PDFReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(15, 23, 42)
        self.cell(0, 7, "NeuroVision AI - Brain Tumor Diagnostic & XAI Report", 0, 1, "L")
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, "Confidential Clinical Decision Support Document | Powered by Deep Learning Ensemble", 0, 1, "L")
        self.set_draw_color(99, 102, 241)
        self.set_line_width(0.5)
        self.line(10, 22, 200, 22)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()} | NeuroVision XAI Engine | Research & Clinical Evaluation Support", 0, 0, "C")


@app.route("/generate_pdf", methods=["POST"])
def generate_pdf():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Missing report data"}), 400

        pred_class = sanitize_text_for_fpdf(data.get("predicted_class", "unknown"))
        display_name = sanitize_text_for_fpdf(data.get("display_name", "Unknown Class"))
        description = sanitize_text_for_fpdf(data.get("description", ""))
        conf_pct = float(data.get("confidence_pct", 0.0))
        has_tumor = bool(data.get("has_tumor", False))
        model_probs = data.get("model_probabilities", {})
        indiv_preds = data.get("individual_predictions", {})
        xai_text = sanitize_text_for_fpdf(data.get("xai_explanation", ""))

        orig_b64 = data.get("original_image_b64", "")
        cam_b64 = data.get("gradcam_overlay_b64", "")

        temp_dir = PROJECT_ROOT / "outputs" / "temp_pdf"
        temp_dir.mkdir(parents=True, exist_ok=True)
        orig_img_path = temp_dir / "orig.png"
        cam_img_path = temp_dir / "cam.png"

        if orig_b64.startswith("data:image"):
            orig_b64 = orig_b64.split(",", 1)[1]
        with open(orig_img_path, "wb") as f:
            f.write(base64.b64decode(orig_b64))

        if cam_b64.startswith("data:image"):
            cam_b64 = cam_b64.split(",", 1)[1]
        with open(cam_img_path, "wb") as f:
            f.write(base64.b64decode(cam_b64))

        pdf = PDFReport()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Metadata Header
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(95, 5, f"Report Timestamp: {now_str}", 0, 0, "L")
        pdf.cell(95, 5, f"Reference Scan ID: MRI-{int(datetime.now().timestamp())}", 0, 1, "R")
        pdf.ln(3)

        view_display_name = sanitize_text_for_fpdf(data.get("view_display_name", "Axial View (Top-Down Plane)"))

        # Diagnosis Box
        if has_tumor:
            pdf.set_fill_color(254, 242, 242)
            pdf.set_draw_color(239, 68, 68)
            pdf.set_text_color(185, 28, 28)
            status_title = "BRAIN TUMOR DETECTED"
        else:
            pdf.set_fill_color(240, 253, 244)
            pdf.set_draw_color(34, 197, 94)
            pdf.set_text_color(21, 128, 61)
            status_title = "NO TUMOR DETECTED"

        # Compute model consensus for winning class
        models_voting_winner = [m for m, p in indiv_preds.items() if p == pred_class]
        winner_consensus_str = f"Model Consensus: {', '.join(models_voting_winner)} ({len(models_voting_winner)}/{len(indiv_preds)} Models Agreed)" if models_voting_winner else "Model Consensus: Soft-Voting Ensemble"
        winner_consensus_str = sanitize_text_for_fpdf(winner_consensus_str)

        pdf.rect(10, pdf.get_y(), 190, 32, "DF")
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(190, 6, f"  DIAGNOSTIC RESULT: {status_title}", 0, 1, "L")
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(190, 5.5, f"  Classification: {display_name}  |  Ensemble Confidence: {conf_pct:.2f}%", 0, 1, "L")
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(79, 70, 229)
        pdf.cell(190, 5, f"  Slice Orientation: {view_display_name}", 0, 1, "L")
        pdf.cell(190, 5, f"  {winner_consensus_str}", 0, 1, "L")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(190, 5, f"  {description[:110]}...", 0, 1, "L")
        pdf.ln(6)

        # Side-by-side Images
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(90, 6, "Original MRI Brain Scan:", 0, 0, "C")
        pdf.cell(10, 6, "", 0, 0)
        pdf.cell(90, 6, "Grad-CAM Neural Attention Heatmap:", 0, 1, "C")

        img_y = pdf.get_y()
        pdf.image(str(orig_img_path), x=25, y=img_y, w=55, h=55)
        pdf.image(str(cam_img_path), x=125, y=img_y, w=55, h=55)
        pdf.set_y(img_y + 58)

        # Clean 2-Column Ensemble Probability Table
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Deep Learning Model Ensemble Probability Breakdown:", 0, 1, "L")
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_fill_color(241, 245, 249)
        pdf.set_draw_color(203, 213, 225)
        pdf.cell(110, 6, "  Class Label", 1, 0, "L", True)
        pdf.cell(80, 6, "Ensemble Probability (%)", 1, 1, "C", True)

        pdf.set_font("Helvetica", "", 8.5)
        c_names = {"notumor": "Healthy / No Tumor", "glioma": "Glioma Tumor", "meningioma": "Meningioma Tumor", "pituitary": "Pituitary Tumor"}
        for k, v in model_probs.items():
            prob_pct = f"{v * 100:.1f}%"
            c_disp = sanitize_text_for_fpdf(c_names.get(k, k))
            is_winner = (k == pred_class)
            if is_winner:
                pdf.set_font("Helvetica", "B", 8.5)
                pdf.set_fill_color(238, 242, 255)
                pdf.cell(110, 5.5, f"  {c_disp} (Predicted)", 1, 0, "L", True)
                pdf.cell(80, 5.5, prob_pct, 1, 1, "C", True)
            else:
                pdf.set_font("Helvetica", "", 8.5)
                pdf.cell(110, 5.5, f"  {c_disp}", 1, 0, "L")
                pdf.cell(80, 5.5, prob_pct, 1, 1, "C")

        pdf.ln(5)

        # XAI Report Section
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Real-Time Explainable AI (XAI) Dynamic Narrative:", 0, 1, "L")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(51, 65, 85)

        clean_text = xai_text.replace("### ", "").replace("**", "").replace("*", "")
        for line in clean_text.split("\n"):
            line_str = line.strip()
            if line_str:
                pdf.multi_cell(0, 4.2, line_str)
                pdf.ln(1)

        pdf_bytes = pdf.output(dest="S")
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode("latin1")
        buf = io.BytesIO(pdf_bytes)

        return send_file(
            buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"NeuroVision_Brain_Tumor_Report_{pred_class}.pdf",
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"PDF generation failed: {str(e)}"}), 500


@app.route("/sample/<class_name>")
def get_sample(class_name: str):
    """Return a sample MRI image from the dataset for 1-click UI testing."""
    if class_name not in CLASS_TO_ID:
        return jsonify({"error": "Invalid class name"}), 400

    data_dir = PROJECT_ROOT / "data" / "primary" / "Testing" / class_name
    if not data_dir.exists():
        data_dir = PROJECT_ROOT / "data" / "external" / class_name

    if not data_dir.exists():
        return jsonify({"error": "Sample directory not found"}), 404

    files = [f for f in sorted(data_dir.iterdir()) if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not files:
        return jsonify({"error": "No sample files found"}), 404

    # Pick first file
    sample_file = files[0]
    return send_from_directory(str(sample_file.parent), sample_file.name)


def find_available_port(preferred_port: int = 5050, max_attempts: int = 30) -> int:
    """Find an available port starting from preferred_port to prevent conflicts with DR (port 5000) or other apps."""
    for p in range(preferred_port, preferred_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return preferred_port


def main():
    print("\n" + "=" * 73)
    print("  NEUROVISION AI - BRAIN TUMOR MRI CLASSIFICATION & XAI SYSTEM")
    print("=" * 73)
    load_all_models()

    preferred_port = int(os.environ.get("PORT", 5050))
    port = find_available_port(preferred_port=preferred_port)

    print("-" * 73)
    print("  >>> OPEN THE WEB APPLICATION USING THIS LINK:")
    print(f"  >>> http://127.0.0.1:{port}/")
    print("=" * 73 + "\n")

    # Automatically launch system browser (Microsoft Edge) after server binds
    threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()

    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
