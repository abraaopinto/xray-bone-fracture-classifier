from __future__ import annotations

import torch
from xray_bone_fracture_classifier.interpretability import GradCAM, overlay_cam_on_pil

import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import streamlit as st
from PIL import Image

from xray_bone_fracture_classifier.inference.predictor import load_bundle, predict_image

st.set_page_config(page_title="Bone Fracture Classifier", layout="centered")

st.title("Bone Fracture Classifier")
st.caption("Upload a bone X-ray image and receive a fracture class prediction (Top-K).")


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _format_float(x: Any, ndigits: int = 4) -> str:
    try:
        if x is None:
            return "N/A"
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return "N/A"


@st.cache_resource
def _load_model_bundle(model_dir: str):
    return load_bundle(Path(model_dir))


def _find_runs(models_root: Path) -> List[Path]:
    if not models_root.exists():
        return []
    runs = [p for p in models_root.iterdir() if p.is_dir() and (p / "model.pt").exists() and (p / "config.json").exists()]
    return sorted(runs)


models_root = Path("models")
reports_root = Path("reports")
run_dirs = _find_runs(models_root)

with st.sidebar:
    st.header("Model")

    if not run_dirs:
        st.error("No trained models found in ./models. Train a model first (scripts/train.py).")
        st.stop()

    selected_run = st.selectbox("Select a trained run", run_dirs, format_func=lambda p: p.name)
    topk = st.slider("Top-K", min_value=1, max_value=5, value=3)

    show_gradcam = st.checkbox("Show Grad-CAM", value=True)
    gradcam_alpha = st.slider("Grad-CAM overlay alpha", 0.10, 0.80, 0.45, 0.05)
    explain_mode = st.radio("Explain class", ["Predicted (Top-1)", "Select from Top-K"], horizontal=False)


    # Load metrics for the selected run, if present
    metrics_path = reports_root / selected_run.name / "metrics.json"
    metrics = _load_json(metrics_path)

    st.divider()
    st.subheader("Test metrics")

    if metrics is None:
        st.warning("No metrics found for this run. Run scripts/evaluate.py to generate reports.")
    else:
        st.metric("Accuracy", _format_float(metrics.get("accuracy")))
        # new schema
        if "f1_macro_present" in metrics or "f1_macro_all" in metrics:
            st.metric("F1 macro (present)", _format_float(metrics.get("f1_macro_present")))
            st.metric("F1 macro (all)", _format_float(metrics.get("f1_macro_all")))
            st.caption(f"Classes in train: {metrics.get('num_classes_all', 'N/A')} | Present in test: {metrics.get('num_classes_present_in_test', 'N/A')}")
        else:
            # legacy fallback
            st.metric("F1 macro", _format_float(metrics.get("f1_macro")))

    st.divider()
    st.caption(f"Model dir: {selected_run.as_posix()}")


bundle = _load_model_bundle(str(selected_run))

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if uploaded is None:
    st.info("Upload an image to run inference.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
st.image(img, caption="Input image", use_container_width=True)

col1, col2 = st.columns([1, 1])
with col1:
    do_predict = st.button("Predict", use_container_width=True)
with col2:
    show_details = st.checkbox("Show details", value=True)

if not do_predict:
    st.stop()

# Use a temporary file to reuse existing predictor without altering its signature
with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
    tmp_path = Path(tmp.name)
    img.save(tmp_path)

try:
    top = predict_image(bundle=bundle, image_path=tmp_path, topk=topk)
finally:
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

pred_label, pred_prob = top[0]
# Map label -> class index using bundle metadata
# (assume bundle has idx_to_class and class_to_idx; adjust if your bundle uses different names)
class_to_idx = getattr(bundle, "class_to_idx", None)
idx_to_class = getattr(bundle, "idx_to_class", None)

if class_to_idx is None or idx_to_class is None:
    # fallback: try reading from bundle.labels (dict)
    labels = getattr(bundle, "labels", None)
    if isinstance(labels, dict):
        # labels likely {str(idx): name} or {idx: name}
        idx_to_class = {int(k): v for k, v in labels.items()}
        class_to_idx = {v: k for k, v in idx_to_class.items()}
    else:
        st.warning("Grad-CAM unavailable: bundle does not expose class mapping.")
        show_gradcam = False

target_label = pred_label
if show_gradcam and explain_mode == "Select from Top-K":
    target_label = st.selectbox("Target label for Grad-CAM", [lbl for lbl, _ in top], index=0)

target_class = class_to_idx.get(target_label) if show_gradcam else None
st.subheader("Prediction")
st.write(f"**{pred_label}**  (p={pred_prob:.4f})")

if show_details:
    st.markdown("### Top-K probabilities")
    for label, prob in top:
        st.write(f"{label} — {prob:.4f}")
        st.progress(min(max(prob, 0.0), 1.0))

if show_gradcam and target_class is not None:
    st.markdown("### Grad-CAM (explanation)")

    # Prepare the same tensor used by the model (reuse bundle preprocessing)
    # Expect bundle to expose a transform callable used in inference.
    transform = getattr(bundle, "transform", None)
    model = getattr(bundle, "model", None)

    if transform is None or model is None:
        st.warning("Grad-CAM unavailable: bundle is missing model/transform.")
    else:
        x = transform(img).unsqueeze(0)  # [1, C, H, W]

        # Grad-CAM requires gradients => ensure model is in eval but grad enabled
        model.eval()
        for p in model.parameters():
            p.requires_grad_(True)

        gradcam = GradCAM(model)
        try:
            res = gradcam(x, target_class=int(target_class))
        finally:
            gradcam.close()

        overlay = overlay_cam_on_pil(img, res.cam, alpha=float(gradcam_alpha))
        st.image(
            overlay,
            caption=f"Grad-CAM for '{target_label}' (p={res.score:.4f})",
            use_container_width=True,
        )
        st.caption(
            "Note: Grad-CAM highlights regions that most influenced the model for the selected class. "
            "It is a qualitative explanation, not a clinical localization."
        )