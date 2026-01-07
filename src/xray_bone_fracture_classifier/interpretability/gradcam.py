from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def _find_last_conv_layer(module: nn.Module) -> nn.Module:
    """
    Find the last Conv2d layer in a model to use as Grad-CAM target.
    Works for common CNNs (e.g., ResNet variants).
    """
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("No Conv2d layer found in model. Grad-CAM requires convolutional features.")
    return last_conv


@dataclass
class GradCAMResult:
    cam: np.ndarray  # HxW in [0, 1]
    target_class: int
    score: float


class GradCAM:
    """
    Minimal Grad-CAM implementation for PyTorch classifiers.

    Usage:
      gradcam = GradCAM(model, target_layer=None)  # auto-detect last conv
      result = gradcam(image_tensor, target_class=None)  # use predicted class
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model.eval()
        self.device = next(model.parameters()).device

        self.target_layer = target_layer if target_layer is not None else _find_last_conv_layer(model)

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def close(self) -> None:
        """Remove hooks."""
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
        if self._bwd_handle is not None:
            self._bwd_handle.remove()

    def _forward_hook(self, module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        self._activations = out.detach()

    def _backward_hook(self, module: nn.Module, grad_in: Tuple[torch.Tensor, ...], grad_out: Tuple[torch.Tensor, ...]) -> None:
        # grad_out[0] is gradient w.r.t. layer output
        self._gradients = grad_out[0].detach()

    @torch.inference_mode(False)
    def __call__(self, x: torch.Tensor, target_class: Optional[int] = None) -> GradCAMResult:
        """
        Compute Grad-CAM for a single image tensor.
        Args:
            x: Tensor [1, C, H, W]
            target_class: class index to explain. If None, uses argmax.
        Returns:
            GradCAMResult with CAM in [0,1].
        """
        x = x.to(self.device)
        self.model.zero_grad(set_to_none=True)
        self._activations = None
        self._gradients = None

        logits = self.model(x)  # [1, num_classes]
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(probs, dim=1).item())

        score = float(probs[0, target_class].item())

        # Backprop on the selected logit (better than prob for Grad-CAM stability)
        selected_logit = logits[0, target_class]
        selected_logit.backward(retain_graph=False)

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients. Check target layer selection.")

        # activations/gradients: [1, K, h, w]
        acts = self._activations
        grads = self._gradients

        # Global average pool the gradients over spatial dims -> weights [1, K, 1, 1]
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # Weighted sum over channels -> [1, 1, h, w]
        cam = (weights * acts).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return GradCAMResult(cam=cam.detach().cpu().numpy(), target_class=target_class, score=score)


def overlay_cam_on_pil(
    image_pil: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Overlay a CAM heatmap on a PIL image.

    Args:
        image_pil: RGB image (original resolution)
        cam: HxW float array in [0,1] (usually model input resolution)
        alpha: blending factor (0-1)

    Returns:
        PIL.Image with heatmap overlay.
    """
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")

    # Resize CAM to original image size (W,H)
    w, h = image_pil.size
    cam_img = Image.fromarray((np.clip(cam, 0.0, 1.0) * 255).astype(np.uint8))
    cam_img = cam_img.resize((w, h), resample=Image.BILINEAR)
    cam_rs = np.array(cam_img).astype(np.float32) / 255.0  # HxW

    img = np.array(image_pil).astype(np.float32) / 255.0  # HxWx3

    # simple "jet-like" colormap (no external deps)
    r = np.clip(1.5 - np.abs(2.0 * cam_rs - 1.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * cam_rs - 0.5), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(2.0 * cam_rs - 0.0), 0.0, 1.0)
    heat = np.stack([r, g, b], axis=-1)  # HxWx3

    out = (1.0 - alpha) * img + alpha * heat
    out = np.clip(out, 0.0, 1.0)

    return Image.fromarray((out * 255).astype(np.uint8))
