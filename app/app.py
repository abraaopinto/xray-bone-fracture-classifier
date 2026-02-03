# app/app.py
from __future__ import annotations

import json
import tempfile

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image

from xray_bone_fracture_classifier.inference.predictor import ModelBundle, load_bundle, predict_image
from xray_bone_fracture_classifier.interpretability import GradCAM, overlay_cam_on_pil


# ----------------------------
# Helpers
# ----------------------------
def list_model_runs(models_root: Path) -> List[Path]:
    """Return model run directories that contain required artifacts."""
    if not models_root.exists():
        return []

    runs = [p for p in models_root.iterdir() if p.is_dir()]
    good: List[Path] = []
    for p in sorted(runs, reverse=True):
        if (p / "model.pt").exists() and (p / "config.json").exists() and (p / "labels.json").exists():
            good.append(p)
    return good


@st.cache_resource(show_spinner=False)
def cached_bundle(model_dir_str: str, device_str: Optional[str]) -> ModelBundle:
    """Cache model bundle to avoid reloading weights at every interaction."""
    model_dir = Path(model_dir_str)
    return load_bundle(model_dir=model_dir, device=device_str)


def save_app_artifacts(
    out_root: Path,
    run_dir: Path,
    image: Image.Image,
    preds: List[Tuple[str, float]],
    gradcam_img: Optional[Image.Image] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist input + predictions (+ optional gradcam) for auditability."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = out_root / f"app_run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = out_dir / "input_image.png"
    image.save(image_path)

    payload: Dict[str, Any] = {
        "timestamp": ts,
        "model_dir": str(run_dir),
        "topk": [{"label": lbl, "prob": float(prob)} for lbl, prob in preds],
    }
    if extra:
        payload["extra"] = extra

    (out_dir / "predictions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if gradcam_img is not None:
        gradcam_path = out_dir / "gradcam.png"
        gradcam_img.save(gradcam_path)

    return out_dir


def safe_open_image(uploaded_file) -> Optional[Image.Image]:
    try:
        return Image.open(uploaded_file).convert("RGB")
    except Exception:
        return None


def _bundle_device_label(bundle: ModelBundle) -> str:
    try:
        return str(bundle.device)
    except Exception:
        return "unknown"


# ----------------------------
# UI
# ----------------------------
def main() -> None:
    st.set_page_config(page_title="X-Ray Bone Fracture Classifier", layout="wide")
    st.title("X-Ray Bone Fracture Classifier")

    models_root = Path("models")
    reports_root = Path("reports") / "app_runs"
    reports_root.mkdir(parents=True, exist_ok=True)

    # Sidebar
    with st.sidebar:
        st.header("Configuração")

        device_choice = st.selectbox(
            "Device",
            options=["auto", "cpu", "cuda", "directml"],
            index=0,
            help="auto = CUDA se disponível; senão CPU. directml requer dependência opcional.",
        )
        device_arg: Optional[str] = None if device_choice == "auto" else device_choice

        topk = st.slider("Top-K", min_value=1, max_value=5, value=3)
        enable_gradcam = st.checkbox("Gerar Grad-CAM", value=True)

        st.divider()

        runs = list_model_runs(models_root)
        if not runs:
            st.error(
                "Nenhum modelo encontrado em `./models`.\n\n"
                "Esperado: pastas de run contendo `model.pt`, `config.json`, `labels.json`."
            )
            st.stop()

        run_dir_str = st.selectbox("Modelo (run)", options=[str(p) for p in runs], index=0)
        st.caption(f"Selecionado: `{run_dir_str}`")

        st.divider()

        if st.button("Limpar", type="secondary"):
            st.session_state.clear()
            st.rerun()

    # Main area: upload
    uploaded = st.file_uploader("Envie uma imagem (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded is None:
        st.info("Selecione um modelo e envie uma imagem para iniciar a predição.")
        return

    img = safe_open_image(uploaded)
    if img is None:
        st.error("Arquivo inválido. Envie uma imagem PNG/JPG válida.")
        return

    # Load model bundle (cached)
    with st.spinner("Carregando modelo..."):
        try:
            bundle = cached_bundle(run_dir_str, device_arg)
        except Exception as e:
            st.error(f"Falha ao carregar o modelo do diretório selecionado.\n\nDetalhes: {e}")
            return

    # Predict
    with st.spinner("Executando predição..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp_path = Path(tmp.name)
                img.save(tmp_path)
            preds = predict_image(bundle=bundle, image_path=tmp_path, topk=int(topk))
        except Exception as e:
            st.error(f"Falha na predição.\n\nDetalhes: {e}")
            return

    # Layout: results
    colA, colB = st.columns([1, 1])

    with colA:
        st.subheader("Imagem")
        st.image(img, use_container_width=True)

    with colB:
        st.subheader("Top-K")
        for i, (label, prob) in enumerate(preds, start=1):
            st.write(f"**{i}. {label}** — {prob:.4f}")

        # progress bar for top-1
        top1_prob = float(preds[0][1]) if preds else 0.0
        st.progress(min(max(top1_prob, 0.0), 1.0))
        st.caption(f"Device em uso: `{_bundle_device_label(bundle)}`")

    # Grad-CAM (optional, non-blocking)
    gradcam_overlay: Optional[Image.Image] = None
    if enable_gradcam:
        st.subheader("Grad-CAM")
        try:
            cam = GradCAM(bundle.model)
            x = bundle.transform(img).unsqueeze(0).to(bundle.device)

            top1_label = preds[0][0]
            class_idx = int(bundle.class_to_idx[top1_label])

            result = cam(x, target_class=class_idx)
            cam.close()

            gradcam_overlay = overlay_cam_on_pil(img, result.cam)
            st.image(gradcam_overlay, use_container_width=True)
        except Exception as e:
            st.warning(f"Grad-CAM não pôde ser gerado: {e}")

    st.divider()

    # Save artifacts
    if st.button("Salvar resultado", type="primary"):
        try:
            out_dir = save_app_artifacts(
                out_root=reports_root,
                run_dir=Path(run_dir_str),
                image=img,
                preds=preds,
                gradcam_img=gradcam_overlay,
                extra={
                    "device_choice": device_choice,
                    "topk": int(topk),
                    "gradcam_enabled": bool(enable_gradcam),
                },
            )
            st.success(f"Resultado salvo em: `{out_dir}`")
        except Exception as e:
            st.error(f"Falha ao salvar artefatos.\n\nDetalhes: {e}")


if __name__ == "__main__":
    main()
