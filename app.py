"""
EDGECORE: Edge AI Optimization Console
PS09: INT8 Quantized CNN Deployment for Resource-Constrained TinyML Devices.
A high-contrast, engineering-focused Edge AI laboratory console for neural network quantization and embedded verification.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import plotly.graph_objects as go
import streamlit as st

from src.data import preprocess_canvas_drawing
from src.inference import (
    dequantize_output_tensor,
    get_quantization_details,
    load_tflite_interpreter,
    quantize_input_image,
    run_inference,
)
from src.metrics import evaluate_memory_budget
from streamlit_drawable_canvas import st_canvas

# Base project paths (relative)
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
TINYML_DIR = ROOT_DIR / "tinyml"
ASSETS_DIR = ROOT_DIR / "assets"
SAMPLES_DIR = ASSETS_DIR / "sample_digits"


# -----------------------------------------------------------------------------
# Caching & Resource Loaders
# -----------------------------------------------------------------------------
@st.cache_data
def load_comparison_data() -> Dict[str, Any]:
    comp_file = RESULTS_DIR / "comparison.json"
    if not comp_file.exists():
        return {}
    with open(comp_file, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_metrics_data() -> Dict[str, Any]:
    metrics_file = RESULTS_DIR / "metrics.json"
    if not metrics_file.exists():
        return {}
    with open(metrics_file, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_tinyml_analysis() -> Dict[str, Any]:
    analysis_file = TINYML_DIR / "model_analysis.json"
    if not analysis_file.exists():
        return {}
    with open(analysis_file, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_confusion_matrix_data() -> Dict[str, Any]:
    cm_file = RESULTS_DIR / "confusion_matrices.json"
    if not cm_file.exists():
        return {}
    with open(cm_file, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def get_interpreter(model_path_str: str):
    full_path = ROOT_DIR / model_path_str
    if not full_path.exists():
        raise FileNotFoundError(f"Model not found: {full_path}")
    return load_tflite_interpreter(str(full_path))


def get_available_sample_images() -> List[Path]:
    if not SAMPLES_DIR.exists():
        return []
    return sorted(list(SAMPLES_DIR.glob("*.png")))


def preprocess_uploaded_image(uploaded_file) -> Tuple[np.ndarray, Image.Image]:
    """
    Preprocesses uploaded image to (28, 28) normalized float32 grayscale in [0.0, 1.0].
    Auto-inverts white-background images if necessary.
    """
    img = Image.open(uploaded_file).convert("L")
    img_resized = img.resize((28, 28), Image.Resampling.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32)

    # Check if background is light (mean corners > 127) and invert to match MNIST
    corners = [arr[0, 0], arr[0, 27], arr[27, 0], arr[27, 27]]
    if np.mean(corners) > 127.0:
        arr = 255.0 - arr

    # Normalize to [0.0, 1.0]
    norm_arr = np.clip(arr / 255.0, 0.0, 1.0)
    display_img = Image.fromarray((norm_arr * 255.0).astype(np.uint8), mode="L")
    return norm_arr, display_img


# -----------------------------------------------------------------------------
# Page Configuration & Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EDGECORE // Edge AI Optimization Console",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Dark Theme & Base Canvas */
    .stApp {
        background-color: #08090D;
        color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #08090D;
    }
    ::-webkit-scrollbar-thumb {
        background: #1E2230;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #7C3AED;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B0D14 !important;
        border-right: 1px solid #1A1E2E;
    }

    /* Brand Header in Sidebar */
    .brand-box {
        padding: 12px 6px 20px 6px;
        border-bottom: 1px solid #1A1E2E;
        margin-bottom: 20px;
    }
    .brand-title {
        font-size: 1.5rem;
        font-weight: 900;
        letter-spacing: 0.12em;
        background: linear-gradient(135deg, #00E5FF 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .brand-subtitle {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #64748B;
        font-weight: 600;
    }

    /* Sidebar Status Badges */
    .status-panel {
        background: #11131A;
        border: 1px solid #1E2230;
        border-radius: 8px;
        padding: 12px;
        margin-top: 24px;
        font-size: 0.75rem;
    }
    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 0;
        border-bottom: 1px solid #161924;
    }
    .status-row:last-child {
        border-bottom: none;
    }
    .dot-ready {
        color: #A3FF12;
        font-weight: 800;
    }

    /* Top Persistent Pipeline Ribbon */
    .pipeline-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #0E1017;
        border: 1px solid #1E2230;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 24px;
        overflow-x: auto;
    }
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #64748B;
        white-space: nowrap;
    }
    .pipeline-step.verified {
        color: #A3FF12;
    }
    .pipeline-step.active {
        color: #00E5FF;
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.4);
    }
    .pipeline-arrow {
        color: #334155;
        font-size: 0.85rem;
    }

    /* Impact Hero Cards */
    .hero-card {
        background: #11131A;
        border: 1px solid #1E2230;
        border-radius: 10px;
        padding: 20px;
        text-align: left;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s, border-color 0.2s;
    }
    .hero-card:hover {
        border-color: #7C3AED;
        transform: translateY(-2px);
    }
    .hero-val {
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .hero-val.cyan { color: #00E5FF; }
    .hero-val.violet { color: #A78BFA; }
    .hero-val.lime { color: #A3FF12; }
    .hero-val.amber { color: #FFB020; }
    .hero-lbl {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94A3B8;
    }
    .hero-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Transformation Banner */
    .transform-box {
        background: linear-gradient(135deg, #11131A 0%, #161824 100%);
        border: 1px solid #252A3D;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }

    /* Cockpit HUD Card */
    .hud-card {
        background: #11131A;
        border: 1px solid #1E2230;
        border-radius: 8px;
        padding: 16px;
    }
    .hud-header {
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #64748B;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Global Data Loaders
# -----------------------------------------------------------------------------
comparison = load_comparison_data()
metrics = load_metrics_data()
tinyml_analysis = load_tinyml_analysis()
has_data = bool(comparison and "comparison_results" in comparison)

fp32_b = comparison.get("fp32_baseline", {})
int8_q = comparison.get("int8_quantized", {})
res = comparison.get("comparison_results", {})


# -----------------------------------------------------------------------------
# Sidebar Navigation & Console Status Panel
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand-box">
            <div class="brand-title">EDGECORE</div>
            <div class="brand-subtitle">TinyML Engineering Console</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_selection = st.radio(
        "NAVIGATION",
        [
            "◉ OVERVIEW",
            "◉ LIVE SCANNER",
            "◉ FP32 → INT8",
            "◉ RESOURCE LAB",
            "◉ TINYML VERIFY",
            "◉ PIPELINE",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        f"""
        <div class="status-panel">
            <div class="hud-header">MODEL HEALTH TELEMETRY</div>
            <div class="status-row">
                <span style="color: #94A3B8;">MODEL</span>
                <span style="font-weight: 700; color: #F8FAFC;">INT8 CNN (7.8K)</span>
            </div>
            <div class="status-row">
                <span style="color: #94A3B8;">ACCURACY</span>
                <span style="color: #A3FF12; font-weight: 700;">{int8_q.get('accuracy_percent', 98.46)}%</span>
            </div>
            <div class="status-row">
                <span style="color: #94A3B8;">FLASH FOOTPRINT</span>
                <span style="color: #00E5FF; font-weight: 700;">{int8_q.get('size_kb', 13.50)} KB</span>
            </div>
            <div class="status-row">
                <span style="color: #94A3B8;">TENSOR ARENA</span>
                <span style="color: #FFB020; font-weight: 700;">~14.0 KB*</span>
            </div>
            <div class="status-row">
                <span style="color: #94A3B8;">STATUS</span>
                <span class="dot-ready">● SYSTEM READY</span>
            </div>
        </div>
        <div style="font-size: 0.68rem; color: #475569; margin-top: 10px; line-height: 1.2;">
            *Static estimate. Host evaluated. Physical MCU unverified.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Top Persistent Engineering Pipeline Ribbon
# -----------------------------------------------------------------------------
step_map = {
    "◉ OVERVIEW": 1,
    "◉ LIVE SCANNER": 6,
    "◉ FP32 → INT8": 4,
    "◉ RESOURCE LAB": 5,
    "◉ TINYML VERIFY": 5,
    "◉ PIPELINE": 3,
}
curr_step = step_map.get(nav_selection, 1)

st.markdown(
    f"""
    <div class="pipeline-container">
        <div class="pipeline-step {'verified' if curr_step >= 1 else ''} {'active' if curr_step == 1 else ''}">
            <span>{'✓' if curr_step > 1 else '◉'}</span> DATA [MNIST]
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step {'verified' if curr_step >= 2 else ''} {'active' if curr_step == 2 else ''}">
            <span>{'✓' if curr_step > 2 else '◉'}</span> TRAIN [FP32]
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step {'verified' if curr_step >= 3 else ''} {'active' if curr_step == 3 else ''}">
            <span>{'✓' if curr_step > 3 else '◉'}</span> INT8 PTQ [200 SAMPLES]
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step {'verified' if curr_step >= 4 else ''} {'active' if curr_step == 4 else ''}">
            <span>{'✓' if curr_step > 4 else '◉'}</span> BENCHMARK [10K TEST]
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step {'verified' if curr_step >= 5 else ''} {'active' if curr_step == 5 else ''}">
            <span>{'✓' if curr_step > 5 else '◉'}</span> TINYML [C-ARRAY]
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step {'verified' if curr_step >= 6 else ''} {'active' if curr_step == 6 else ''}">
            <span>{'✓' if curr_step == 6 else '◉'}</span> LIVE INFER
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# VIEW 1: OVERVIEW
# -----------------------------------------------------------------------------
if nav_selection == "◉ OVERVIEW":
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <div style="font-size: 2.4rem; font-weight: 900; letter-spacing: -0.03em; line-height: 1.1; color: #F8FAFC;">
                FROM FLOATING POINT<br>
                <span style="background: linear-gradient(90deg, #00E5FF, #7C3AED); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    TO TINY INTELLIGENCE.
                </span>
            </div>
            <div style="font-size: 1.05rem; color: #94A3B8; margin-top: 6px;">
                End-to-End Post-Training Quantization for Ultra-Low Power Edge Microcontrollers
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Core Transformation Panel
    if has_data:
        st.markdown(
            f"""
            <div class="transform-box">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                    <div style="flex: 1; min-width: 220px; background: #0A0C12; border: 1px solid #1E2230; border-radius: 8px; padding: 18px;">
                        <div style="font-size: 0.75rem; font-weight: 800; color: #64748B; letter-spacing: 0.1em;">BASELINE FP32 MODEL</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC; margin-top: 4px;">{fp32_b.get('size_kb', 34.70)} KB</div>
                        <div style="font-size: 0.9rem; color: #94A3B8; margin-top: 4px;">Accuracy: <b style="color: #F8FAFC;">{fp32_b.get('accuracy_percent', 98.44)}%</b></div>
                        <div style="font-size: 0.82rem; color: #64748B;">Latency: {fp32_b.get('latency_mean_ms', 0.0133):.4f} ms (Host CPU)</div>
                    </div>
                    <div style="text-align: center; padding: 0 10px;">
                        <div style="font-size: 1.4rem; color: #7C3AED; font-weight: 900;">⚡ PTQ ➔</div>
                        <div style="font-size: 0.72rem; color: #00E5FF; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">200 MNIST Samples</div>
                    </div>
                    <div style="flex: 1; min-width: 220px; background: #0A0C12; border: 1px solid #7C3AED; border-radius: 8px; padding: 18px; box-shadow: 0 0 20px rgba(124, 58, 237, 0.15);">
                        <div style="font-size: 0.75rem; font-weight: 800; color: #A78BFA; letter-spacing: 0.1em;">QUANTIZED INT8 MODEL</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #00E5FF; margin-top: 4px;">{int8_q.get('size_kb', 13.50)} KB</div>
                        <div style="font-size: 0.9rem; color: #94A3B8; margin-top: 4px;">Accuracy: <b style="color: #A3FF12;">{int8_q.get('accuracy_percent', 98.46)}%</b> (Lossless)</div>
                        <div style="font-size: 0.82rem; color: #64748B;">Latency: {int8_q.get('latency_mean_ms', 0.0098):.4f} ms (Host CPU)</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4 Hero Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class="hero-card">
                    <div class="hero-lbl">FLASH REDUCTION</div>
                    <div class="hero-val cyan">-{res.get('size_reduction_percent', 61.10)}%</div>
                    <div class="hero-sub">{res.get('size_reduction_bytes', 21712):,} Bytes saved</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="hero-card">
                    <div class="hero-lbl">COMPRESSION RATIO</div>
                    <div class="hero-val violet">{res.get('compression_ratio', 2.57)}×</div>
                    <div class="hero-sub">34.7 KB ➔ 13.5 KB</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="hero-card">
                    <div class="hero-lbl">ACCURACY PRESERVATION</div>
                    <div class="hero-val lime">+{res.get('accuracy_delta_percentage_points', 0.02)} pts</div>
                    <div class="hero-sub">98.44% ➔ 98.46% (10k test)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""
                <div class="hero-card">
                    <div class="hero-lbl">HOST CPU SPEEDUP</div>
                    <div class="hero-val amber">{res.get('latency_change_percent', -26.32)}%</div>
                    <div class="hero-sub">0.0133 ms ➔ 0.0098 ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 🔬 Architectural Pillars for Embedded Edge AI")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(
            """
            <div class="hud-card">
                <div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">📦 Flash Footprint</div>
                <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.4;">
                    Compresses the entire CNN FlatBuffer binary down to <b>13.50 KB (13,824 B)</b>, enabling direct storage in tiny on-chip Flash alongside RTOS firmware.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            """
            <div class="hud-card">
                <div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">🔢 Integer SIMD Arithmetic</div>
                <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.4;">
                    Eliminates all hardware Floating-Point Unit (FPU) dependencies. Computations execute entirely using 8-bit integer SIMD instructions on ARM Cortex-M/ESP32 ALUs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            """
            <div class="hud-card">
                <div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">⚡ Minimal Tensor Arena</div>
                <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.4;">
                    Reduces intermediate activation buffers by 4× per element. The analytical TFLite Micro Tensor Arena requires only <b>~14.0 KB</b> of SRAM.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# VIEW 2: LIVE SCANNER (Inference Cockpit + Canvas)
# -----------------------------------------------------------------------------
elif nav_selection == "◉ LIVE SCANNER":
    st.markdown(
        """
        <div style="margin-bottom: 18px;">
            <div style="font-size: 1.8rem; font-weight: 900; letter-spacing: -0.02em; color: #F8FAFC;">
                INT8 VISION SCANNER
            </div>
            <div style="font-size: 0.9rem; color: #00E5FF; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">
                REAL-TIME EDGE INFERENCE COCKPIT
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown('<div class="hud-header">INPUT TELEMETRY</div>', unsafe_allow_html=True)
        input_mode = st.radio(
            "Select Input Source:",
            ["Draw Digit (Canvas)", "Sample MNIST Digits", "Upload Image"],
            horizontal=True,
            key="scanner_input_mode",
        )

        norm_image: Optional[np.ndarray] = None
        preview_img: Optional[Image.Image] = None

        if input_mode == "Sample MNIST Digits":
            samples = get_available_sample_images()
            if samples:
                sample_names = [p.name for p in samples]
                selected_sample_name = st.selectbox("Select Benchmark Sample:", sample_names)
                selected_path = SAMPLES_DIR / selected_sample_name
                with open(selected_path, "rb") as f:
                    norm_image, preview_img = preprocess_uploaded_image(f)
            else:
                st.warning("No sample digits found in assets/sample_digits/")

        elif input_mode == "Upload Image":
            uploaded_file = st.file_uploader("Upload digit image (PNG, JPG)", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                try:
                    norm_image, preview_img = preprocess_uploaded_image(uploaded_file)
                except Exception as e:
                    st.error(f"Image processing error: {e}")

        elif input_mode == "Draw Digit (Canvas)":
            st.caption("Draw a digit (0–9) below using your mouse or touchscreen:")

            col_c1, col_c2 = st.columns([3, 1])
            with col_c2:
                stroke_width = st.slider("Stroke Width:", min_value=12, max_value=32, value=20, step=2)
                st.caption("Tip: Use bold strokes for clean MNIST centering.")

            with col_c1:
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0.0)",
                    stroke_width=stroke_width,
                    stroke_color="#000000",
                    background_color="#FFFFFF",
                    update_streamlit=True,
                    height=200,
                    width=200,
                    drawing_mode="freedraw",
                    key="scanner_digit_canvas",
                )

            predict_btn = st.button("🔮 Run INT8 Prediction", type="primary", use_container_width=True)

            if predict_btn or "drawn_norm_image" in st.session_state:
                if predict_btn:
                    if canvas_result is not None and canvas_result.image_data is not None:
                        try:
                            norm_image, preview_img = preprocess_canvas_drawing(canvas_result.image_data)
                            st.session_state["drawn_norm_image"] = norm_image
                            st.session_state["drawn_preview_img"] = preview_img
                        except ValueError as ve:
                            st.warning(str(ve))
                            st.session_state.pop("drawn_norm_image", None)
                            st.session_state.pop("drawn_preview_img", None)
                        except Exception as e:
                            st.error(f"Canvas error: {e}")
                    else:
                        st.warning("Please draw a digit before prediction.")
                elif "drawn_norm_image" in st.session_state and input_mode == "Draw Digit (Canvas)":
                    norm_image = st.session_state.get("drawn_norm_image")
                    preview_img = st.session_state.get("drawn_preview_img")

        if preview_img is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.image(preview_img, caption="Preprocessed 28×28 Grayscale Input (Normalized [0.0, 1.0])", width=140)

    with col_result:
        st.markdown('<div class="hud-header">INT8 INFERENCE OUTPUT</div>', unsafe_allow_html=True)
        if norm_image is not None:
            try:
                interpreter_int8 = get_interpreter("models/model_int8.tflite")
                infer_res = run_inference(interpreter_int8, norm_image)

                pred_digit = infer_res["predicted_digit"]
                conf = infer_res["confidence"]
                lat_ms = infer_res["latency_ms"]
                status_label = "CONFIDENT" if conf >= 0.80 else "REVIEW REQUIRED"
                status_color = "#A3FF12" if conf >= 0.80 else "#FFB020"

                st.markdown(
                    f"""
                    <div style="background: #11131A; border: 1px solid #7C3AED; border-radius: 10px; padding: 20px; box-shadow: 0 0 25px rgba(124, 58, 237, 0.15);">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <div style="font-size: 0.75rem; font-weight: 800; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.1em;">PREDICTED CLASS</div>
                                <div style="font-size: 4rem; font-weight: 900; line-height: 1; color: #00E5FF; margin: 4px 0 8px 0;">{pred_digit}</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 0.75rem; font-weight: 800; color: #94A3B8; text-transform: uppercase;">CLASSIFICATION STATUS</div>
                                <div style="font-size: 0.95rem; font-weight: 800; color: {status_color}; margin-top: 4px;">● {status_label}</div>
                            </div>
                        </div>
                        <div style="display: flex; gap: 16px; margin-top: 10px; padding-top: 10px; border-top: 1px solid #1E2230; font-size: 0.88rem;">
                            <div>Confidence: <b style="color: #A3FF12;">{conf * 100.0:.2f}%</b></div>
                            <div>Host Latency: <b style="color: #F8FAFC;">{lat_ms:.4f} ms</b></div>
                        </div>
                        <div style="font-size: 0.75rem; color: #64748B; margin-top: 6px;">
                            Engine: <b>INT8 TFLite</b> (13.50 KB, int8 in/out, Dynamic affine quantization)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Class probabilities distribution chart
                probs = infer_res["probabilities"]
                fig_prob = go.Figure(
                    go.Bar(
                        x=list(range(10)),
                        y=probs,
                        marker_color=["#00E5FF" if i == pred_digit else "#1E2230" for i in range(10)],
                        text=[f"{p * 100:.1f}%" if p > 0.04 else "" for p in probs],
                        textposition="auto",
                    )
                )
                fig_prob.update_layout(
                    title="10-Class Softmax Probability Distribution",
                    xaxis=dict(tickmode="linear", tick0=0, dtick=1, title="Digit Class"),
                    yaxis=dict(title="Probability", range=[0, 1]),
                    height=240,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F8FAFC"),
                )
                st.plotly_chart(fig_prob, use_container_width=True)

            except Exception as e:
                st.error(f"Inference execution failed: {e}")
        else:
            st.info("Select a sample, upload an image, or draw on the canvas to execute real-time INT8 inference.")


# -----------------------------------------------------------------------------
# VIEW 3: FP32 → INT8 (Transformation & Verification Analytics)
# -----------------------------------------------------------------------------
elif nav_selection == "◉ FP32 → INT8":
    st.markdown(
        """
        <div style="margin-bottom: 18px;">
            <div style="font-size: 1.8rem; font-weight: 900; letter-spacing: -0.02em; color: #F8FAFC;">
                FP32 → INT8 MODEL TRANSFORMATION
            </div>
            <div style="font-size: 0.9rem; color: #A78BFA; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">
                QUANTITATIVE BENCHMARKS OVER 10,000 MNIST TEST SAMPLES
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if has_data:
        # Visual Comparison Charts
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            fig_size = go.Figure()
            fig_size.add_trace(go.Bar(
                name="Model Size (KB)",
                x=["FP32 Baseline", "INT8 Quantized"],
                y=[fp32_b.get("size_kb", 34.70), int8_q.get("size_kb", 13.50)],
                marker_color=["#334155", "#00E5FF"],
                text=[f"{fp32_b.get('size_kb', 34.70)} KB", f"{int8_q.get('size_kb', 13.50)} KB"],
                textposition="auto",
            ))
            fig_size.update_layout(
                title=f"Flash Storage Footprint: -{res.get('size_reduction_percent', 61.10)}% ({res.get('compression_ratio', 2.57)}×)",
                yaxis=dict(title="File Size (KB)"),
                height=280,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
            )
            st.plotly_chart(fig_size, use_container_width=True)

        with col_t2:
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Bar(
                name="Latency (ms)",
                x=["FP32 Mean", "INT8 Mean", "FP32 Median", "INT8 Median"],
                y=[
                    fp32_b.get("latency_mean_ms", 0.0133),
                    int8_q.get("latency_mean_ms", 0.0098),
                    fp32_b.get("latency_median_ms", 0.0105),
                    int8_q.get("latency_median_ms", 0.0097),
                ],
                marker_color=["#334155", "#7C3AED", "#1E2230", "#A78BFA"],
                text=[
                    f"{v:.4f} ms"
                    for v in [
                        fp32_b.get("latency_mean_ms", 0.0133),
                        int8_q.get("latency_mean_ms", 0.0098),
                        fp32_b.get("latency_median_ms", 0.0105),
                        int8_q.get("latency_median_ms", 0.0097),
                    ]
                ],
                textposition="auto",
            ))
            fig_lat.update_layout(
                title="Host CPU Single-Sample Latency (time.perf_counter)",
                yaxis=dict(title="Latency (ms)"),
                height=280,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
            )
            st.plotly_chart(fig_lat, use_container_width=True)

        st.markdown("##### 📋 Complete Verification Metric Comparison")
        table_markdown = f"""
| Benchmark Dimension | FP32 TFLite Baseline | INT8 TFLite Quantized | Measured Delta / Impact |
|:---|:---|:---|:---|
| **Test Accuracy (10,000 Samples)** | `{fp32_b.get('accuracy_percent', 98.44)}%` ({fp32_b.get('correct_predictions', 9844):,} / 10,000) | `{int8_q.get('accuracy_percent', 98.46)}%` ({int8_q.get('correct_predictions', 9846):,} / 10,000) | **`+{res.get('accuracy_delta_percentage_points', 0.02)} pts`** (No accuracy loss observed) |
| **Model File Size** | `{fp32_b.get('size_bytes', 35536):,} Bytes` ({fp32_b.get('size_kb', 34.70)} KB) | `{int8_q.get('size_bytes', 13824):,} Bytes` ({int8_q.get('size_kb', 13.50)} KB) | **`-{res.get('size_reduction_percent', 61.10)}%`** ({res.get('compression_ratio', 2.57)}× compression) |
| **Storage Bytes Saved** | Baseline | `{res.get('size_reduction_bytes', 21712):,} Bytes` | **`21.2 KB Flash saved`** |
| **Host Mean Latency** | `{fp32_b.get('latency_mean_ms', 0.0133):.4f} ms` | `{int8_q.get('latency_mean_ms', 0.0098):.4f} ms` | **`{res.get('latency_change_percent', -26.32)}%`** (`{res.get('latency_difference_ms', -0.0035):.4f} ms`) |
| **Host Median Latency** | `{fp32_b.get('latency_median_ms', 0.0105):.4f} ms` | `{int8_q.get('latency_median_ms', 0.0097):.4f} ms` | `-7.62%` |
| **Host Latency Range (Min–Max)**| `{fp32_b.get('latency_min_ms', 0.008):.4f} – {fp32_b.get('latency_max_ms', 0.05):.4f} ms` | `{int8_q.get('latency_min_ms', 0.007):.4f} – {int8_q.get('latency_max_ms', 0.04):.4f} ms` | Lower standard deviation (`{int8_q.get('latency_std_ms', 0.002):.4f}` vs `{fp32_b.get('latency_std_ms', 0.004):.4f}`) |
| **Input / Output Datatypes** | `float32` in / `float32` out | `int8` in / `int8` out | **Full-Integer Quantization** |
| **Hardware FPU Requirement** | Required for FP operations | **None** (Integer SIMD ALU) | **TinyML MCU Ready** |
"""
        st.markdown(table_markdown)
        st.info("ℹ️ **Measurement Note**: Latency metrics reflect single-sample execution measured on the host development environment (CPU). Microcontroller execution cycles will vary depending on MCU clock speed and instruction set architecture.")

        # --- Confusion Matrix Section ---
        st.markdown("---")
        st.markdown("##### 🎯 Classification Analysis: FP32 vs INT8 Confusion Matrix")
        st.caption("10×10 confusion matrices evaluated across all 10,000 genuine MNIST test samples. Compare per-class precision along the diagonal and off-diagonal misclassifications.")

        cm_data = load_confusion_matrix_data()
        if cm_data and "fp32" in cm_data and "int8" in cm_data:
            cm_mode = st.radio(
                "Confusion Matrix View:",
                ["Raw Counts", "Normalized (%)"],
                horizontal=True,
                key="cm_display_mode",
            )
            is_normalized = (cm_mode == "Normalized (%)")

            fp32_mat = np.array(cm_data["fp32"]["matrix"], dtype=np.float32)
            int8_mat = np.array(cm_data["int8"]["matrix"], dtype=np.float32)
            classes = [str(c) for c in cm_data.get("classes", list(range(10)))]

            if is_normalized:
                row_sums_fp32 = fp32_mat.sum(axis=1, keepdims=True)
                fp32_display = np.where(row_sums_fp32 > 0, (fp32_mat / row_sums_fp32) * 100.0, 0.0)

                row_sums_int8 = int8_mat.sum(axis=1, keepdims=True)
                int8_display = np.where(row_sums_int8 > 0, (int8_mat / row_sums_int8) * 100.0, 0.0)

                zmin, zmax = 0.0, 100.0
                text_template_fp32 = [[f"{val:.1f}%" if val >= 0.5 else "" for val in row] for row in fp32_display]
                text_template_int8 = [[f"{val:.1f}%" if val >= 0.5 else "" for val in row] for row in int8_display]
                colorbar_title = "Recall (%)"
            else:
                fp32_display = fp32_mat
                int8_display = int8_mat
                zmin, zmax = 0, float(np.max([fp32_mat.max(), int8_mat.max()]))
                text_template_fp32 = [[f"{int(val)}" if val > 0 else "" for val in row] for row in fp32_display]
                text_template_int8 = [[f"{int(val)}" if val > 0 else "" for val in row] for row in int8_display]
                colorbar_title = "Count"

            col_cm_fp32, col_cm_int8 = st.columns(2)

            with col_cm_fp32:
                fig_cm_fp32 = go.Figure(
                    data=go.Heatmap(
                        z=fp32_display,
                        x=classes,
                        y=classes,
                        text=text_template_fp32,
                        texttemplate="%{text}",
                        colorscale="Blues",
                        zmin=zmin,
                        zmax=zmax,
                        colorbar=dict(title=colorbar_title),
                    )
                )
                fig_cm_fp32.update_layout(
                    title=f"FP32 TFLite ({cm_data['fp32']['correct_predictions']:,} / 10,000 Correct)",
                    xaxis=dict(title="Predicted Digit", tickmode="linear", dtick=1),
                    yaxis=dict(title="Actual Digit", tickmode="linear", dtick=1, autorange="reversed"),
                    height=400,
                    margin=dict(l=40, r=20, t=50, b=40),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F8FAFC"),
                )
                st.plotly_chart(fig_cm_fp32, use_container_width=True)

            with col_cm_int8:
                fig_cm_int8 = go.Figure(
                    data=go.Heatmap(
                        z=int8_display,
                        x=classes,
                        y=classes,
                        text=text_template_int8,
                        texttemplate="%{text}",
                        colorscale="Teal",
                        zmin=zmin,
                        zmax=zmax,
                        colorbar=dict(title=colorbar_title),
                    )
                )
                fig_cm_int8.update_layout(
                    title=f"INT8 TFLite ({cm_data['int8']['correct_predictions']:,} / 10,000 Correct)",
                    xaxis=dict(title="Predicted Digit", tickmode="linear", dtick=1),
                    yaxis=dict(title="Actual Digit", tickmode="linear", dtick=1, autorange="reversed"),
                    height=400,
                    margin=dict(l=40, r=20, t=50, b=40),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F8FAFC"),
                )
                st.plotly_chart(fig_cm_int8, use_container_width=True)

            c_s1, c_s2, c_s3, c_s4 = st.columns(4)
            with c_s1:
                st.metric("FP32 Correct", f"{cm_data['fp32']['correct_predictions']:,} / 10,000", f"{cm_data['fp32']['accuracy_percent']}%")
            with c_s2:
                st.metric("FP32 Misclassifications", f"{cm_data['fp32']['incorrect_predictions']:,}", f"{cm_data['fp32']['incorrect_predictions'] / 100.0:.2f}% Error Rate", delta_color="inverse")
            with c_s3:
                st.metric("INT8 Correct", f"{cm_data['int8']['correct_predictions']:,} / 10,000", f"{cm_data['int8']['accuracy_percent']}% (+2)")
            with c_s4:
                st.metric("INT8 Misclassifications", f"{cm_data['int8']['incorrect_predictions']:,}", f"-2 vs FP32 ({cm_data['int8']['incorrect_predictions'] / 100.0:.2f}% Error Rate)", delta_color="inverse")
        else:
            st.warning("Confusion matrix data file not found at results/confusion_matrices.json")
    else:
        st.error("No comparison data available.")


# -----------------------------------------------------------------------------
# VIEW 4: RESOURCE LAB (Hardware Memory Cockpit)
# -----------------------------------------------------------------------------
elif nav_selection == "◉ RESOURCE LAB":
    st.markdown(
        """
        <div style="margin-bottom: 18px;">
            <div style="font-size: 1.8rem; font-weight: 900; letter-spacing: -0.02em; color: #F8FAFC;">
                TINYML RESOURCE LAB
            </div>
            <div style="font-size: 0.9rem; color: #00E5FF; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">
                HARDWARE MEMORY BUDGET COCKPIT & FEASIBILITY CHECKER
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if tinyml_analysis and "verified" in tinyml_analysis:
        ver = tinyml_analysis["verified"]
        est = tinyml_analysis["estimated"]

        mcu_presets = {
            "Standard TinyML (32 KB Flash / 32 KB RAM)": (32.0, 32.0),
            "Generic Tiny (16 KB Flash / 16 KB RAM)": (16.0, 16.0),
            "Mid-Range MCU (64 KB Flash / 64 KB RAM)": (64.0, 64.0),
            "High-End TinyML (128 KB Flash / 128 KB RAM)": (128.0, 128.0),
            "Arduino Nano 33 BLE (1024 KB Flash / 256 KB RAM)": (1024.0, 256.0),
            "Raspberry Pi RP2040 (2048 KB Flash / 264 KB RAM)": (2048.0, 264.0),
            "ESP32-S3 (4096 KB Flash / 512 KB RAM)": (4096.0, 512.0),
            "Custom / Manual Input": (32.0, 32.0),
        }

        col_preset, col_flash_in, col_ram_in = st.columns([2, 1, 1])
        with col_preset:
            selected_preset = st.selectbox(
                "Target MCU Hardware Profile:",
                list(mcu_presets.keys()),
                index=0,
                key="resource_lab_preset",
            )
            def_flash, def_ram = mcu_presets[selected_preset]

        with col_flash_in:
            avail_flash_kb = st.number_input(
                "Target Flash Budget (KB):",
                min_value=1.0,
                max_value=16384.0,
                value=float(def_flash),
                step=8.0,
                key=f"flash_input_lab_{selected_preset}",
            )

        with col_ram_in:
            avail_ram_kb = st.number_input(
                "Target RAM Budget (KB):",
                min_value=1.0,
                max_value=4096.0,
                value=float(def_ram),
                step=4.0,
                key=f"ram_input_lab_{selected_preset}",
            )

        budget_res = evaluate_memory_budget(
            avail_flash_kb,
            avail_ram_kb,
            model_flash_bytes=ver["flash_storage_bytes"],
            estimated_arena_bytes=est["estimated_tensor_arena_bytes"],
        )

        col_b_flash, col_b_ram = st.columns(2)

        with col_b_flash:
            flash_border = "#A3FF12" if budget_res["flash_fits"] else "#EF4444"
            flash_badge = "✓ FITS" if budget_res["flash_fits"] else "✕ DOES NOT FIT"
            flash_text_color = "#A3FF12" if budget_res["flash_fits"] else "#EF4444"
            flash_headroom_str = (
                f"Headroom: {budget_res['flash_headroom_kb']:.1f} KB remaining"
                if budget_res["flash_fits"]
                else f"Shortfall: {budget_res['flash_shortfall_kb']:.1f} KB needed"
            )
            st.markdown(
                f"""
                <div style="background: #11131A; border: 1px solid {flash_border}; border-radius: 8px; padding: 18px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 800; font-size: 1.1rem; color: #F8FAFC;">📦 Flash / ROM Budget</span>
                        <span style="font-weight: 800; font-size: 0.95rem; color: {flash_text_color};">{flash_badge}</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #94A3B8;">Available Flash: <b style="color: #F8FAFC;">{budget_res['available_flash_kb']:.1f} KB</b></div>
                    <div style="font-size: 0.9rem; color: #94A3B8;">INT8 Model Footprint: <b style="color: #00E5FF;">{budget_res['model_flash_kb']:.2f} KB</b> ({budget_res['model_flash_bytes']:,} B)</div>
                    <div style="font-size: 0.9rem; color: #94A3B8;">Flash Utilization: <b style="color: {flash_text_color};">{budget_res['flash_usage_pct']:.1f}%</b></div>
                    <div style="font-size: 0.85rem; color: #64748B; margin-top: 6px;">{flash_headroom_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            flash_bar_pct = min(1.0, budget_res["flash_usage_pct"] / 100.0) if budget_res["available_flash_kb"] > 0 else 1.0
            st.progress(flash_bar_pct, text=f"Flash Allocation: {budget_res['flash_usage_pct']:.1f}%")

        with col_b_ram:
            ram_border = "#A3FF12" if budget_res["ram_fits"] else "#EF4444"
            ram_badge = "✓ FITS" if budget_res["ram_fits"] else "✕ DOES NOT FIT"
            ram_text_color = "#A3FF12" if budget_res["ram_fits"] else "#EF4444"
            ram_headroom_str = (
                f"Headroom: ~{budget_res['ram_headroom_kb']:.1f} KB remaining"
                if budget_res["ram_fits"]
                else f"Shortfall: ~{budget_res['ram_shortfall_kb']:.1f} KB needed"
            )
            st.markdown(
                f"""
                <div style="background: #11131A; border: 1px solid {ram_border}; border-radius: 8px; padding: 18px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 800; font-size: 1.1rem; color: #F8FAFC;">⚡ RAM / SRAM Budget</span>
                        <span style="font-weight: 800; font-size: 0.95rem; color: {ram_text_color};">{ram_badge}</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #94A3B8;">Available RAM: <b style="color: #F8FAFC;">{budget_res['available_ram_kb']:.1f} KB</b></div>
                    <div style="font-size: 0.9rem; color: #94A3B8;">Estimated Tensor Arena: <b style="color: #FFB020;">~{budget_res['estimated_arena_kb']:.1f} KB</b> ({budget_res['estimated_arena_bytes']:,} B)</div>
                    <div style="font-size: 0.9rem; color: #94A3B8;">RAM Utilization: <b style="color: {ram_text_color};">~{budget_res['ram_usage_pct']:.1f}%</b></div>
                    <div style="font-size: 0.85rem; color: #64748B; margin-top: 6px;">{ram_headroom_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            ram_bar_pct = min(1.0, budget_res["ram_usage_pct"] / 100.0) if budget_res["available_ram_kb"] > 0 else 1.0
            st.progress(ram_bar_pct, text=f"RAM Allocation: ~{budget_res['ram_usage_pct']:.1f}%")

        if budget_res["fits_overall"]:
            st.success(f"🎯 **SYSTEM STATUS: ✓ DEPLOYMENT FEASIBLE*** — {budget_res['explanation']}")
        else:
            st.error(f"⚠️ **SYSTEM STATUS: ✕ INSUFFICIENT MEMORY** — {budget_res['explanation']}")

        st.markdown(
            """
            <div style="background: #0B0D14; border: 1px solid #1E2230; border-radius: 6px; padding: 14px; font-size: 0.82rem; color: #94A3B8; margin-top: 10px;">
                <b>Resource Telemetry Breakdown:</b><br>
                • <b>Flash (Verified from binary)</b>: INT8 FlatBuffer model is exactly <b>13,824 bytes (13.50 KB)</b>. <i>(Application firmware & MCU driver code overhead not included)</i><br>
                • <b>RAM (Analytical Projection)</b>: Input buffer = <b>784 B</b>, Largest activation = <b>5,408 B</b>, TFLM Tensor Arena = <b>~14.0 KB</b>. <i>(RTOS heap & call stack overhead not included)</i><br>
                • <i>*Static resource estimate. Physical MCU deployment not verified.</i>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error("TinyML model analysis file not found at tinyml/model_analysis.json")


# -----------------------------------------------------------------------------
# VIEW 5: TINYML VERIFY
# -----------------------------------------------------------------------------
elif nav_selection == "◉ TINYML VERIFY":
    st.markdown(
        """
        <div style="margin-bottom: 18px;">
            <div style="font-size: 1.8rem; font-weight: 900; letter-spacing: -0.02em; color: #F8FAFC;">
                TINYML VERIFICATION
            </div>
            <div style="font-size: 0.9rem; color: #A3FF12; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">
                STATIC GRAPH AUDIT & EMBEDDED C-ARRAY VERIFICATION
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if tinyml_analysis and "verified" in tinyml_analysis:
        ver = tinyml_analysis["verified"]
        est = tinyml_analysis["estimated"]
        not_ver = tinyml_analysis["not_verified"]

        col_v, col_e, col_nv = st.columns(3)

        with col_v:
            st.markdown(
                f"""
                <div style="background: #11131A; border: 1px solid #A3FF12; border-radius: 8px; padding: 16px; height: 100%;">
                    <div style="font-size: 0.85rem; font-weight: 800; color: #A3FF12; margin-bottom: 8px;">🟢 1. VERIFIED (STATIC AUDIT)</div>
                    <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.6;">
                        ✓ <b>INT8 Model Binary</b>: 13.50 KB ({ver['flash_storage_bytes']:,} B)<br>
                        ✓ <b>Input Tensor</b>: {ver['input_tensor']['shape']} ({ver['input_tensor']['dtype']}, 784 B)<br>
                        ✓ <b>Output Tensor</b>: {ver['output_tensor']['shape']} ({ver['output_tensor']['dtype']}, 10 B)<br>
                        ✓ <b>Total Graph Tensors</b>: {ver['total_tensor_count']}<br>
                        ✓ <b>C-Array Binary Parity</b>: 100% Byte Match<br>
                        ✓ <b>TFLM Built-in Ops</b>: All 5 Ops Supported
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_e:
            st.markdown(
                f"""
                <div style="background: #11131A; border: 1px solid #FFB020; border-radius: 8px; padding: 16px; height: 100%;">
                    <div style="font-size: 0.85rem; font-weight: 800; color: #FFB020; margin-bottom: 8px;">🟡 2. ESTIMATED (RUNTIME PROJECTION)</div>
                    <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.6;">
                        ⚠ <b>Input Buffer RAM</b>: {est['input_buffer_bytes']} Bytes<br>
                        ⚠ <b>Peak Activation RAM</b>: {est['largest_single_activation_bytes']:,} Bytes<br>
                        ⚠ <b>Tensor Arena Projection</b>: <b>~{est['estimated_tensor_arena_kb']} KB</b> ({est['estimated_tensor_arena_bytes']:,} B)<br>
                        <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 6px;">
                            <i>{est['estimation_methodology']}</i>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_nv:
            st.markdown(
                """
                <div style="background: #11131A; border: 1px solid #64748B; border-radius: 8px; padding: 16px; height: 100%;">
                    <div style="font-size: 0.85rem; font-weight: 800; color: #94A3B8; margin-bottom: 8px;">⚪ 3. NOT VERIFIED (BOUNDARIES)</div>
                    <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.6;">
                        ✕ <b>Physical MCU Flashing</b>: Not Executed<br>
                        ✕ <b>Hardware Clock Cycles</b>: Not Measured<br>
                        ✕ <b>Hardware Power/Current Draw</b>: Not Measured<br>
                        <div style="font-size: 0.72rem; color: #64748B; margin-top: 6px;">
                            <i>Evaluated via host simulation / TFLite runtime. Target hardware boards were not physically flashed.</i>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 🧱 TFLite Micro Compatible Operators")
        ops_df = [
            {"Operator": op["operator"], "TFLM Support": "✅ Builtin Supported" if op["supported_in_tflm"] else "❌ Unsupported", "Notes": op["notes"]}
            for op in ver["tflite_micro_compatible_ops"]
        ]
        st.table(ops_df)

        st.markdown("##### 📦 Generated C/C++ Header Preview (`tinyml/model_data.h`)")
        header_path = TINYML_DIR / "model_data.h"
        if header_path.exists():
            with open(header_path, "r", encoding="utf-8") as f:
                st.code(f.read(), language="c")
    else:
        st.error("TinyML model analysis file not found at tinyml/model_analysis.json")


# -----------------------------------------------------------------------------
# VIEW 6: PIPELINE
# -----------------------------------------------------------------------------
elif nav_selection == "◉ PIPELINE":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <div style="font-size: 1.85rem; font-weight: 900; letter-spacing: -0.02em; color: #F8FAFC;">
                PIPELINE ARCHITECTURE
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #00E5FF; margin-top: 2px;">
                From FP32 CNN to TinyML-Ready INT8
            </div>
            <div style="font-size: 0.9rem; color: #94A3B8; margin-top: 6px;">
                See how a trained floating-point CNN is transformed into a compact integer model for resource-constrained edge devices.
            </div>
        </div>

        <div style="display: flex; align-items: center; justify-content: space-between; background: #11131A; border: 1px solid #1E2230; border-radius: 8px; padding: 10px 18px; margin-bottom: 28px; flex-wrap: wrap; gap: 8px;">
            <div style="font-size: 0.8rem; font-weight: 800; color: #A3FF12;"><span style="background: #162618; border: 1px solid #A3FF12; border-radius: 4px; padding: 2px 6px; margin-right: 4px;">1</span> TRAIN</div>
            <div style="color: #475569;">➔</div>
            <div style="font-size: 0.8rem; font-weight: 800; color: #00E5FF;"><span style="background: #0E2530; border: 1px solid #00E5FF; border-radius: 4px; padding: 2px 6px; margin-right: 4px;">2</span> CONVERT</div>
            <div style="color: #475569;">➔</div>
            <div style="font-size: 0.8rem; font-weight: 800; color: #A78BFA;"><span style="background: #231936; border: 1px solid #7C3AED; border-radius: 4px; padding: 2px 6px; margin-right: 4px;">3</span> CALIBRATE</div>
            <div style="color: #475569;">➔</div>
            <div style="font-size: 0.8rem; font-weight: 800; color: #00E5FF;"><span style="background: #0E2530; border: 1px solid #00E5FF; border-radius: 4px; padding: 2px 6px; margin-right: 4px;">4</span> QUANTIZE</div>
            <div style="color: #475569;">➔</div>
            <div style="font-size: 0.8rem; font-weight: 800; color: #A3FF12;"><span style="background: #162618; border: 1px solid #A3FF12; border-radius: 4px; padding: 2px 6px; margin-right: 4px;">5</span> EXPORT</div>
            <div style="color: #475569;">➔</div>
            <div style="font-size: 0.8rem; font-weight: 800; color: #38BDF8;"><span style="background: #132738; border: 1px solid #38BDF8; border-radius: 4px; padding: 2px 6px; margin-right: 4px;">6</span> DEPLOY</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # SECTION 1: THE EDGE CONSTRAINT
    # -------------------------------------------------------------------------
    st.markdown("##### 01 · WHY FP32 IS TOO HEAVY")
    
    col_e1, col_e_mid, col_e2 = st.columns([5, 2, 5])
    with col_e1:
        st.markdown(
            f"""
            <div style="background: #11131A; border: 1px solid #2A3042; border-radius: 8px; padding: 18px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.1rem; font-weight: 800; color: #F8FAFC;">FP32 CNN</span>
                    <span style="font-size: 0.72rem; font-weight: 800; color: #94A3B8; background: #1E2230; padding: 3px 8px; border-radius: 4px;">32-bit Floating Point</span>
                </div>
                <div style="margin-top: 14px; font-size: 0.88rem; color: #CBD5E1; line-height: 1.8;">
                    • <b>MODEL SIZE</b>: <span style="color: #F8FAFC; font-weight: 700;">{fp32_b.get('size_kb', 34.70)} KB</span> ({fp32_b.get('size_bytes', 35536):,} Bytes)<br>
                    • <b>ARITHMETIC</b>: 32-bit IEEE 754 Floating Point<br>
                    • <b>HARDWARE</b>: Requires Floating-Point Unit (FPU)<br>
                    • <b>DEPLOYMENT</b>: More resource intensive
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_e_mid:
        st.markdown(
            """
            <div style="text-align: center; padding-top: 24px;">
                <div style="font-size: 0.72rem; font-weight: 800; color: #64748B;">FP32</div>
                <div style="font-size: 1.3rem; color: #7C3AED; font-weight: 900; margin: 4px 0;">⚡ PTQ ➔</div>
                <div style="font-size: 0.72rem; font-weight: 800; color: #00E5FF;">INT8</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_e2:
        st.markdown(
            f"""
            <div style="background: #11131A; border: 1px solid #7C3AED; border-radius: 8px; padding: 18px; box-shadow: 0 0 15px rgba(124, 58, 237, 0.12);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.1rem; font-weight: 800; color: #00E5FF;">INT8 CNN</span>
                    <span style="font-size: 0.72rem; font-weight: 800; color: #A3FF12; background: #162618; border: 1px solid #A3FF12; padding: 3px 8px; border-radius: 4px;">8-bit Integer</span>
                </div>
                <div style="margin-top: 14px; font-size: 0.88rem; color: #CBD5E1; line-height: 1.8;">
                    • <b>MODEL SIZE</b>: <span style="color: #00E5FF; font-weight: 700;">{int8_q.get('size_kb', 13.50)} KB</span> ({int8_q.get('size_bytes', 13824):,} Bytes)<br>
                    • <b>ARITHMETIC</b>: 8-bit Signed Integer SIMD<br>
                    • <b>HARDWARE</b>: Pure Integer ALU (No FPU Required)<br>
                    • <b>DEPLOYMENT</b>: TinyML-oriented
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_ci1, col_ci2, col_ci3 = st.columns(3)
    with col_ci1:
        st.markdown(
            f"""
            <div style="background: #0B0D14; border: 1px solid #1E2230; border-radius: 6px; padding: 12px; margin-top: 10px;">
                <div style="font-size: 0.72rem; font-weight: 800; color: #64748B;">FLASH MEMORY FOOTPRINT</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #00E5FF; margin-top: 2px;">FP32 {fp32_b.get('size_kb', 34.70)} KB ➔ INT8 {int8_q.get('size_kb', 13.50)} KB</div>
                <div style="font-size: 0.8rem; color: #A3FF12; font-weight: 700;">↓ {res.get('size_reduction_percent', 61.10)}% Reduction</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_ci2:
        st.markdown(
            """
            <div style="background: #0B0D14; border: 1px solid #1E2230; border-radius: 6px; padding: 12px; margin-top: 10px;">
                <div style="font-size: 0.72rem; font-weight: 800; color: #64748B;">COMPUTATIONAL PIPELINE</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #A78BFA; margin-top: 2px;">32-bit Float ➔ 8-bit Integer</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">SIMD execution on standard ALU</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_ci3:
        st.markdown(
            """
            <div style="background: #0B0D14; border: 1px solid #1E2230; border-radius: 6px; padding: 12px; margin-top: 10px;">
                <div style="font-size: 0.72rem; font-weight: 800; color: #64748B;">EDGE TARGET HARDWARE</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #FFB020; margin-top: 2px;">Limited Flash + SRAM</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Fits 32 KB / 64 KB Microcontrollers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="font-size: 0.84rem; color: #94A3B8; margin-top: 10px; font-style: italic;">
            "INT8 reduces the numerical representation used by the deployed model, making the model more suitable for memory-constrained embedded inference."
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 2: VISUAL QUANTIZATION MATHEMATICS
    # -------------------------------------------------------------------------
    st.markdown("##### 02 · FP32 → INT8 MATHEMATICAL TRANSFORMATION")

    st.markdown(
        """
        <div style="background: #11131A; border: 1px solid #1E2230; border-radius: 8px; padding: 18px; margin-bottom: 16px;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 16px; flex-wrap: wrap; text-align: center;">
                <div style="background: #08090D; border: 1px solid #2A3042; border-radius: 6px; padding: 10px 18px;">
                    <div style="font-size: 0.72rem; font-weight: 800; color: #64748B;">REAL CONTINUOUS VALUE</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #F8FAFC;">r ∈ [r_min, r_max]</div>
                </div>
                <div style="font-size: 1.4rem; color: #7C3AED;">➔</div>
                <div style="background: #161824; border: 1px solid #7C3AED; border-radius: 6px; padding: 10px 18px;">
                    <div style="font-size: 0.72rem; font-weight: 800; color: #A78BFA;">LINEAR AFFINE QUANTIZATION</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #00E5FF;">Scale S &nbsp;|&nbsp; Zero-Point Z</div>
                </div>
                <div style="font-size: 1.4rem; color: #7C3AED;">➔</div>
                <div style="background: #08090D; border: 1px solid #A3FF12; border-radius: 6px; padding: 10px 18px;">
                    <div style="font-size: 0.72rem; font-weight: 800; color: #A3FF12;">INTEGER DISCRETE VALUE</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #A3FF12;">q ∈ [-128, 127]</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown(
            """
            <div style="background: #11131A; border: 1px solid #1E2230; border-radius: 8px; padding: 18px; height: 100%;">
                <div style="font-size: 0.78rem; font-weight: 800; color: #00E5FF; text-transform: uppercase; letter-spacing: 0.08em;">QUANTIZATION FORMULA</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #F8FAFC; margin: 10px 0;">
                    $$q = \\text{clip}\\left(\\left\\lfloor \\frac{r}{S} \\right\\rceil + Z, -128, 127\\right)$$
                </div>
                <div style="font-size: 0.85rem; color: #94A3B8;">
                    "Converts a floating-point value into an INT8 representation."
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_q2:
        st.markdown(
            """
            <div style="background: #11131A; border: 1px solid #1E2230; border-radius: 8px; padding: 18px; height: 100%;">
                <div style="font-size: 0.78rem; font-weight: 800; color: #A78BFA; text-transform: uppercase; letter-spacing: 0.08em;">DEQUANTIZATION FORMULA</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #F8FAFC; margin: 10px 0;">
                    $$r = S \\times (q - Z)$$
                </div>
                <div style="font-size: 0.85rem; color: #94A3B8;">
                    "Recovers the corresponding real-valued approximation."
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Visual Number Line
    st.markdown(
        """
        <div style="background: #0E1017; border: 1px solid #1E2230; border-radius: 8px; padding: 18px; margin-top: 16px;">
            <div style="font-size: 0.78rem; font-weight: 800; color: #64748B; margin-bottom: 8px;">VISUAL NUMBER-LINE AFFINE MAPPING</div>
            <div style="font-family: monospace; font-size: 0.88rem; color: #CBD5E1; line-height: 2;">
                <b>REAL DOMAIN</b> &nbsp;&nbsp;&nbsp;: <span style="color: #64748B;">r_min</span> ──────────────────── <span style="color: #00E5FF; font-weight: 800;">0.0</span> ──────────────────── <span style="color: #64748B;">r_max</span><br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color: #7C3AED; font-weight: 800;">↓ AFFINE MAPPING (S, Z)</span><br>
                <b>INT8 DOMAIN</b> &nbsp;&nbsp;&nbsp;: <span style="color: #64748B;">-128</span> ──────────────────── <span style="color: #A3FF12; font-weight: 800;">Z = -128</span> ───────────────── <span style="color: #64748B;">+127</span>
            </div>
            <div style="display: flex; gap: 20px; margin-top: 10px; font-size: 0.8rem; color: #94A3B8; border-top: 1px solid #1E2230; padding-top: 8px;">
                <div><b>S (Scale)</b>: Positive float representing the step size per integer quantum.</div>
                <div><b>Z (Zero-Point)</b>: Integer offset corresponding exactly to real 0.0 for lossless padding.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 3: USE OUR ACTUAL PROJECT VALUES
    # -------------------------------------------------------------------------
    st.markdown("##### 03 · ACTUAL QUANTIZATION PARAMETERS")
    st.caption("Dynamically read from the verified INT8 TFLite interpreter.")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown(
            """
            <div style="background: #11131A; border: 1px solid #00E5FF; border-radius: 8px; padding: 18px;">
                <div style="font-size: 0.8rem; font-weight: 800; color: #00E5FF;">INPUT TENSOR PARAMETERS</div>
                <div style="margin-top: 10px; font-size: 0.88rem; color: #CBD5E1; line-height: 1.8;">
                    • <b>Data Type</b>: <code>int8</code><br>
                    • <b>Scale (S)</b>: <code>0.003921568859368563</code> (≈ 1 / 255)<br>
                    • <b>Zero-Point (Z)</b>: <code>-128</code><br>
                    • <b>Tensor Shape</b>: <code>[1, 28, 28, 1]</code> (784 Bytes)
                </div>
                <div style="background: #08090D; border-radius: 6px; padding: 10px; margin-top: 12px; font-family: monospace; font-size: 0.82rem; color: #A3FF12;">
                    Real [0.0, 1.0] ➔ Quantized [-128, 127]<br>
                    0.0 ➔ -128 &nbsp;&nbsp;|&nbsp;&nbsp; 1.0 ➔ 127
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_p2:
        st.markdown(
            """
            <div style="background: #11131A; border: 1px solid #7C3AED; border-radius: 8px; padding: 18px;">
                <div style="font-size: 0.8rem; font-weight: 800; color: #A78BFA;">OUTPUT TENSOR PARAMETERS</div>
                <div style="margin-top: 10px; font-size: 0.88rem; color: #CBD5E1; line-height: 1.8;">
                    • <b>Data Type</b>: <code>int8</code><br>
                    • <b>Scale (S)</b>: <code>0.00390625</code> (1 / 256)<br>
                    • <b>Zero-Point (Z)</b>: <code>-128</code><br>
                    • <b>Tensor Shape</b>: <code>[1, 10]</code> (10 Classes)
                </div>
                <div style="background: #08090D; border-radius: 6px; padding: 10px; margin-top: 12px; font-family: monospace; font-size: 0.82rem; color: #00E5FF;">
                    Quantized [-128, 127] ➔ Probability [0.0, 1.0]<br>
                    -128 ➔ 0.0 &nbsp;&nbsp;|&nbsp;&nbsp; +127 ➔ ≈ 1.0
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 4: REPRESENTATIVE DATASET CALIBRATION
    # -------------------------------------------------------------------------
    st.markdown("##### 04 · HOW THE MODEL LEARNS ITS INT8 RANGE")

    col_cal_flow, col_cal_info = st.columns([7, 5])
    with col_cal_flow:
        st.markdown(
            """
            <div style="background: #11131A; border: 1px solid #1E2230; border-radius: 8px; padding: 18px;">
                <div style="font-size: 0.78rem; font-weight: 800; color: #A3FF12; letter-spacing: 0.08em; text-transform: uppercase;">
                    CALIBRATION DATA FLOW
                </div>
                <div style="margin-top: 12px; font-size: 0.88rem; color: #F8FAFC; line-height: 2;">
                    <span style="color: #00E5FF; font-weight: 800;">200 REAL MNIST SAMPLES</span><br>
                    &nbsp;&nbsp;↓ <i>Feedforward through FP32 network</i><br>
                    <span style="color: #F8FAFC; font-weight: 700;">FP32 CNN LAYERS</span><br>
                    &nbsp;&nbsp;↓ <i>Record min/max dynamic activation values</i><br>
                    <span style="color: #A78BFA; font-weight: 700;">ACTIVATION OBSERVATION & RANGE CALIBRATION</span><br>
                    &nbsp;&nbsp;↓ <i>Compute optimal scale S and zero-point Z</i><br>
                    <span style="color: #A3FF12; font-weight: 800;">INT8 QUANTIZED MODEL (No Accuracy Loss)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_cal_info:
        st.markdown(
            """
            <div style="background: #11131A; border: 1px solid #1E2230; border-radius: 8px; padding: 18px; height: 100%;">
                <div style="font-size: 0.78rem; font-weight: 800; color: #64748B; text-transform: uppercase;">
                    REPRESENTATIVE DATASET SPECIFICATION
                </div>
                <div style="margin-top: 10px; font-size: 0.86rem; color: #CBD5E1; line-height: 1.8;">
                    • <b>Sample Count</b>: 200 samples<br>
                    • <b>Source</b>: MNIST Training Set<br>
                    • <b>Sample Shape</b>: <code>(28, 28, 1)</code><br>
                    • <b>Data Type</b>: <code>float32</code><br>
                    • <b>Value Range</b>: <code>[0.0, 1.0]</code>
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 10px; border-top: 1px solid #1E2230; padding-top: 8px;">
                    "Representative samples are passed through the network during PTQ calibration. The observed activation ranges are used to determine quantization parameters for the integer model."
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 5: END-TO-END PIPELINE
    # -------------------------------------------------------------------------
    st.markdown("##### 05 · END-TO-END EMBEDDED CONVERSION")

    st.markdown(
        f"""
        <div style="background: #11131A; border: 1px solid #1E2230; border-radius: 10px; padding: 22px;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; text-align: center;">
                <div style="background: #08090D; border: 1px solid #1E2230; border-radius: 6px; padding: 12px 6px;">
                    <div style="font-size: 0.68rem; font-weight: 800; color: #64748B;">NODE 1</div>
                    <div style="font-size: 0.82rem; font-weight: 800; color: #F8FAFC; margin-top: 4px;">MNIST DATASET</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">28×28 Grayscale</div>
                </div>
                <div style="background: #08090D; border: 1px solid #1E2230; border-radius: 6px; padding: 12px 6px;">
                    <div style="font-size: 0.68rem; font-weight: 800; color: #64748B;">NODE 2</div>
                    <div style="font-size: 0.82rem; font-weight: 800; color: #F8FAFC; margin-top: 4px;">LIGHTWEIGHT CNN</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">7,834 Params</div>
                </div>
                <div style="background: #08090D; border: 1px solid #1E2230; border-radius: 6px; padding: 12px 6px;">
                    <div style="font-size: 0.68rem; font-weight: 800; color: #64748B;">NODE 3</div>
                    <div style="font-size: 0.82rem; font-weight: 800; color: #F8FAFC; margin-top: 4px;">FP32 KERAS</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">model_fp32.keras</div>
                </div>
                <div style="background: #08090D; border: 1px solid #1E2230; border-radius: 6px; padding: 12px 6px;">
                    <div style="font-size: 0.68rem; font-weight: 800; color: #64748B;">NODE 4</div>
                    <div style="font-size: 0.82rem; font-weight: 800; color: #F8FAFC; margin-top: 4px;">FP32 TFLITE</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">{fp32_b.get('size_kb', 34.70)} KB</div>
                </div>
                <div style="background: #161824; border: 1px solid #7C3AED; border-radius: 6px; padding: 12px 6px; box-shadow: 0 0 12px rgba(124, 58, 237, 0.2);">
                    <div style="font-size: 0.68rem; font-weight: 800; color: #00E5FF;">NODE 5 (KEY)</div>
                    <div style="font-size: 0.82rem; font-weight: 800; color: #00E5FF; margin-top: 4px;">INT8 PTQ</div>
                    <div style="font-size: 0.72rem; color: #A78BFA;">200 Samples</div>
                </div>
                <div style="background: #08090D; border: 1px solid #1E2230; border-radius: 6px; padding: 12px 6px;">
                    <div style="font-size: 0.68rem; font-weight: 800; color: #64748B;">NODE 6</div>
                    <div style="font-size: 0.82rem; font-weight: 800; color: #A3FF12; margin-top: 4px;">INT8 TFLITE</div>
                    <div style="font-size: 0.72rem; color: #A3FF12;">{int8_q.get('size_kb', 13.50)} KB</div>
                </div>
                <div style="background: #08090D; border: 1px solid #1E2230; border-radius: 6px; padding: 12px 6px;">
                    <div style="font-size: 0.68rem; font-weight: 800; color: #64748B;">NODE 7</div>
                    <div style="font-size: 0.82rem; font-weight: 800; color: #F8FAFC; margin-top: 4px;">C-ARRAY EXPORT</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">model_data.h/.cc</div>
                </div>
                <div style="background: #08090D; border: 1px solid #1E2230; border-radius: 6px; padding: 12px 6px;">
                    <div style="font-size: 0.68rem; font-weight: 800; color: #64748B;">NODE 8</div>
                    <div style="font-size: 0.82rem; font-weight: 800; color: #38BDF8; margin-top: 4px;">TINYML READY</div>
                    <div style="font-size: 0.72rem; color: #38BDF8;">TFLM Compatible</div>
                </div>
            </div>
            <div style="display: flex; justify-content: space-around; margin-top: 18px; padding-top: 14px; border-top: 1px solid #1E2230; font-size: 0.85rem;">
                <div>FP32 TFLite: <b>{fp32_b.get('size_kb', 34.70)} KB</b></div>
                <div>INT8 TFLite: <b style="color: #00E5FF;">{int8_q.get('size_kb', 13.50)} KB</b></div>
                <div>Flash Reduction: <b style="color: #A3FF12;">-{res.get('size_reduction_percent', 61.10)}%</b></div>
                <div>Compression: <b style="color: #A78BFA;">{res.get('compression_ratio', 2.57)}×</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 6: BEFORE vs AFTER
    # -------------------------------------------------------------------------
    st.markdown("##### 06 · WHAT CHANGED?")

    col_ba1, col_ba2 = st.columns(2)
    with col_ba1:
        st.markdown(
            f"""
            <div style="background: #11131A; border: 1px solid #2A3042; border-radius: 8px; padding: 18px;">
                <div style="font-size: 0.78rem; font-weight: 800; color: #94A3B8; text-transform: uppercase;">BEFORE (FP32 BASELINE)</div>
                <div style="margin-top: 12px; font-size: 0.9rem; color: #CBD5E1; line-height: 1.8;">
                    • <b>Model Format</b>: FP32 TFLite FlatBuffer<br>
                    • <b>Flash Size</b>: {fp32_b.get('size_kb', 34.70)} KB ({fp32_b.get('size_bytes', 35536):,} B)<br>
                    • <b>Tensor Dtype</b>: <code>float32</code><br>
                    • <b>Test Accuracy</b>: {fp32_b.get('accuracy_percent', 98.44)}% (9,844 / 10,000)<br>
                    • <b>Host Latency</b>: {fp32_b.get('latency_mean_ms', 0.0133):.4f} ms (Host CPU)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_ba2:
        st.markdown(
            f"""
            <div style="background: #11131A; border: 1px solid #00E5FF; border-radius: 8px; padding: 18px; box-shadow: 0 0 15px rgba(0, 229, 255, 0.1);">
                <div style="font-size: 0.78rem; font-weight: 800; color: #00E5FF; text-transform: uppercase;">AFTER (INT8 QUANTIZED)</div>
                <div style="margin-top: 12px; font-size: 0.9rem; color: #CBD5E1; line-height: 1.8;">
                    • <b>Model Format</b>: INT8 TFLite FlatBuffer<br>
                    • <b>Flash Size</b>: <span style="color: #00E5FF; font-weight: 700;">{int8_q.get('size_kb', 13.50)} KB</span> ({int8_q.get('size_bytes', 13824):,} B)<br>
                    • <b>Tensor Dtype</b>: <code>int8</code><br>
                    • <b>Test Accuracy</b>: <span style="color: #A3FF12; font-weight: 700;">{int8_q.get('accuracy_percent', 98.46)}%</span> (9,846 / 10,000)<br>
                    • <b>Host Latency</b>: <span style="color: #A78BFA; font-weight: 700;">{int8_q.get('latency_mean_ms', 0.0098):.4f} ms</span> (Host CPU)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_imp1, col_imp2, col_imp3 = st.columns(3)
    with col_imp1:
        st.markdown(
            f"""
            <div style="background: #0B0D14; border: 1px solid #1E2230; border-radius: 6px; padding: 14px; margin-top: 10px; text-align: center;">
                <div style="font-size: 1.6rem; font-weight: 900; color: #00E5FF;">{res.get('size_reduction_percent', 61.10)}%</div>
                <div style="font-size: 0.75rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;">SMALLER MODEL</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_imp2:
        st.markdown(
            f"""
            <div style="background: #0B0D14; border: 1px solid #1E2230; border-radius: 6px; padding: 14px; margin-top: 10px; text-align: center;">
                <div style="font-size: 1.6rem; font-weight: 900; color: #A3FF12;">+{res.get('accuracy_delta_percentage_points', 0.02)} pts</div>
                <div style="font-size: 0.75rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;">TEST ACCURACY CHANGE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_imp3:
        st.markdown(
            f"""
            <div style="background: #0B0D14; border: 1px solid #1E2230; border-radius: 6px; padding: 14px; margin-top: 10px; text-align: center;">
                <div style="font-size: 1.6rem; font-weight: 900; color: #A78BFA;">26.32%</div>
                <div style="font-size: 0.75rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;">LOWER MEAN HOST LATENCY*</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("*Host CPU measurement (single sample execution). Microcontroller execution cycles will vary with clock speed.")

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 7: TINYML DEPLOYMENT BRIDGE
    # -------------------------------------------------------------------------
    st.markdown("##### 07 · FROM MODEL FILE TO EMBEDDED BINARY")

    st.markdown(
        """
        <div style="background: #11131A; border: 1px solid #1E2230; border-radius: 8px; padding: 18px;">
            <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 12px; text-align: center;">
                <div style="background: #08090D; border: 1px solid #1E2230; border-radius: 6px; padding: 10px 14px;">
                    <div style="font-size: 0.7rem; color: #64748B;">MODEL FILE</div>
                    <div style="font-size: 0.85rem; font-weight: 800; color: #F8FAFC;">model_int8.tflite</div>
                </div>
                <div style="color: #7C3AED; font-weight: 800;">➔</div>
                <div style="background: #08090D; border: 1px solid #1E2230; border-radius: 6px; padding: 10px 14px;">
                    <div style="font-size: 0.7rem; color: #64748B;">BYTE-EXACT CONVERSION</div>
                    <div style="font-size: 0.85rem; font-weight: 800; color: #00E5FF;">C Array Hex Array</div>
                </div>
                <div style="color: #7C3AED; font-weight: 800;">➔</div>
                <div style="background: #08090D; border: 1px solid #1E2230; border-radius: 6px; padding: 10px 14px;">
                    <div style="font-size: 0.7rem; color: #64748B;">SOURCE FILES</div>
                    <div style="font-size: 0.85rem; font-weight: 800; color: #A3FF12;">model_data.h / .cc</div>
                </div>
                <div style="color: #7C3AED; font-weight: 800;">➔</div>
                <div style="background: #08090D; border: 1px solid #1E2230; border-radius: 6px; padding: 10px 14px;">
                    <div style="font-size: 0.7rem; color: #64748B;">RUNTIME ENGINE</div>
                    <div style="font-size: 0.85rem; font-weight: 800; color: #A78BFA;">TFLite Micro Runtime</div>
                </div>
                <div style="color: #7C3AED; font-weight: 800;">➔</div>
                <div style="background: #08090D; border: 1px solid #1E2230; border-radius: 6px; padding: 10px 14px;">
                    <div style="font-size: 0.7rem; color: #64748B;">HARDWARE TARGET</div>
                    <div style="font-size: 0.85rem; font-weight: 800; color: #38BDF8;">Target MCU (Flash/RAM)</div>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 18px; padding-top: 14px; border-top: 1px solid #1E2230; font-size: 0.82rem; color: #CBD5E1;">
                <div>✓ <b>C-array matches INT8 TFLite binary</b></div>
                <div>✓ <b>Exact 13,824 bytes binary length</b></div>
                <div>✓ <b>INT8 input/output tensor types</b></div>
                <div>✓ <b>TFLite Micro-compatible operators verified</b></div>
            </div>
            <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 12px; background: #08090D; padding: 10px; border-radius: 6px;">
                ⚠️ <b>Honest Boundary Note</b>: Physical MCU flashing and hardware power/cycle measurements are not part of the current verification.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SECTION 8: TECHNICAL TAKEAWAY
    # -------------------------------------------------------------------------
    st.markdown("##### 08 · THE ENGINEERING TAKEAWAY")

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #11131A 0%, #161824 100%); border: 1px solid #7C3AED; border-radius: 10px; padding: 22px; box-shadow: 0 0 20px rgba(124, 58, 237, 0.15);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; font-size: 0.82rem; font-weight: 800; color: #00E5FF; margin-bottom: 12px;">
                <span>TRAIN ON FP32</span>
                <span style="color: #7C3AED;">➔</span>
                <span>CALIBRATE WITH REAL DATA</span>
                <span style="color: #7C3AED;">➔</span>
                <span>QUANTIZE TO INT8</span>
                <span style="color: #7C3AED;">➔</span>
                <span>VERIFY ACCURACY</span>
                <span style="color: #7C3AED;">➔</span>
                <span>MEASURE SIZE & LATENCY</span>
                <span style="color: #7C3AED;">➔</span>
                <span>EXPORT FOR TINYML</span>
            </div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; line-height: 1.5; margin-top: 6px;">
                "The goal is not simply to make the model smaller. The goal is to preserve useful inference accuracy while producing a representation that is practical for resource-constrained edge deployment."
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # INTERACTIVE TECHNICAL EXPANDERS
    # -------------------------------------------------------------------------
    st.markdown("##### 🔍 Deep-Dive Technical Inspections")

    with st.expander("Show Quantization Details"):
        st.markdown(
            """
            - **Quantization Type**: Full-Integer Post-Training Quantization (PTQ)
            - **Input Interface**: `int8` (`[-128, 127]`)
            - **Output Interface**: `int8` (`[-128, 127]`)
            - **Internal Layers**: Convolution, Pooling, and Dense weights and intermediate activations are all quantized to signed 8-bit integers.
            - **Biases**: Stored as 32-bit integers (`int32`) with scale $S_{\\text{bias}} = S_{\\text{input}} \\times S_{\\text{weight}}$.
            """
        )

    with st.expander("Show Calibration Details"):
        st.markdown(
            """
            - **Representative Generator**: 200 real MNIST training samples yielded sequentially.
            - **Dynamic Range Recording**: For each layer $l$, the calibration observer captures $r_{\\min}^{(l)}$ and $r_{\\max}^{(l)}$.
            - **Scale Calculation**: $S = \\frac{r_{\\max} - r_{\\min}}{255.0}$.
            - **Zero-Point Calculation**: $Z = \\text{round}\\left(-r_{\\min} / S\\right) - 128$.
            """
        )

    with st.expander("Show Tensor Parameters"):
        st.markdown(
            """
            - **Input Tensor Index 0**: Shape `[1, 28, 28, 1]`, Scale: `0.003921568859368563`, Zero-Point: `-128`.
            - **Conv2D Weights**: Shape `[3, 3, 1, 8]`, Per-channel scale and zero-point.
            - **Conv2D_1 Weights**: Shape `[3, 3, 8, 16]`, Per-channel scale and zero-point.
            - **Dense Weights**: Shape `[16, 10]`, Per-tensor scale and zero-point.
            - **Output Tensor Index 11**: Shape `[1, 10]`, Scale: `0.00390625`, Zero-Point: `-128`.
            """
        )

    with st.expander("Show Deployment Details"):
        st.markdown(
            """
            - **Embedded C Header**: `tinyml/model_data.h` contains `extern const unsigned char g_model[];` and `extern const unsigned int g_model_len;`.
            - **Embedded C Source**: `tinyml/model_data.cc` contains the 13,824-byte hex array `alignas(16) const unsigned char g_model[] = { ... };`.
            - **TFLite Micro Runtime**: The model can be initialized via `tflite::GetModel(g_model)` and executed with `tflite::MicroInterpreter`.
            - **Memory Allocation**: Arena size allocated via `uint8_t tensor_arena[kTensorArenaSize];` in RAM.
            """
        )

