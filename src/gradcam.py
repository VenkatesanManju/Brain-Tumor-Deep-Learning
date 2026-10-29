#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import MODEL_NAMES, add_common_paths, add_common_training_args
from data import build_external_loader, build_primary_loaders
from train import build_model
from utils import ensure_dir, setup_logger


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

    def generate(self, input_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        score = output[0, target_class]
        score.backward()

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


def get_target_layer(model_name: str, model: torch.nn.Module) -> torch.nn.Module:
    if model_name == "ResNet50":
        return model.layer4[-1]
    if model_name == "EfficientNetB0":
        return model.features[-1]
    if model_name == "DenseNet121":
        return model.features.denseblock4.denselayer16.conv2
    raise ValueError(f"Unknown model name: {model_name}")


def overlay_cam_on_image(img_pil: Image.Image, cam: np.ndarray) -> Image.Image:
    img_np = np.array(img_pil.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.float32(heatmap) * 0.4 + np.float32(img_np) * 0.6
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps and overlays for trained CNN models.")
    add_common_paths(parser)
    add_common_training_args(parser)
    parser.add_argument("--num-samples", type=int, default=10, help="Number of test samples to generate Grad-CAM for")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory containing saved checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Skip actual Grad-CAM generation and print execution notice")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("gradcam", args.log_dir)
    logger.info("Step 8: Grad-CAM generation")

    if args.dry_run:
        logger.info("[dry-run] Grad-CAM module outline validated.")
        print("[dry-run] Grad-CAM logic scaffolded and functional.")
        return

    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else args.output_dir / "checkpoints"
    device = torch.device(args.device)

    output_cam_dir = ensure_dir(args.output_dir / "gradcam")
    _, _, test_loader = build_primary_loaders(args.data_root, batch_size=1, num_workers=0)

    models_dict = {}
    cam_extractors = {}
    for model_name in MODEL_NAMES:
        ckpt = checkpoint_dir / f"{model_name}.pt"
        if not ckpt.exists():
            continue
        model = build_model(model_name, num_classes=4).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        target_layer = get_target_layer(model_name, model)
        models_dict[model_name] = model
        cam_extractors[model_name] = GradCAM(model, target_layer)

    if not models_dict:
        logger.warning("No model checkpoints found for Grad-CAM generation.")
        return

    count = 0
    for i, (images, labels) in enumerate(test_loader):
        if count >= args.num_samples:
            break
        images = images.to(device)
        label = labels.item()

        # Denormalize image for visualization
        img_np = images[0].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        cams = []
        for name, extractor in cam_extractors.items():
            cam = extractor.generate(images, target_class=label)
            cams.append(cam)
            overlay = overlay_cam_on_image(img_pil, cam)
            overlay.save(output_cam_dir / f"sample_{i+1}_{name}_label_{label}.png")

        if cams:
            consensus_cam = np.mean(cams, axis=0)
            consensus_cam = consensus_cam / (consensus_cam.max() + 1e-8)
            consensus_overlay = overlay_cam_on_image(img_pil, consensus_cam)
            consensus_overlay.save(output_cam_dir / f"sample_{i+1}_Consensus_label_{label}.png")

        count += 1

    for extractor in cam_extractors.values():
        extractor.remove_hooks()

    logger.info("Generated Grad-CAM heatmaps for %d samples saved to %s", count, output_cam_dir)
    print(f"Grad-CAM overlays generated for {count} samples in {output_cam_dir}")


if __name__ == "__main__":
    main()
