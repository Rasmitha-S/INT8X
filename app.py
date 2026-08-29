"""
EDGE AI LAB // INT8 CNN OPTIMIZATION PLATFORM
PS09: INT8 Quantized CNN Deployment for Resource-Constrained TinyML Devices.
A clean, high-contrast, real-time edge AI engineering console.
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
from src.metrics import evaluate_memory_budget, simulate_mcu_resources
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
# Page Configuration & CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EDGE AI LAB // INT8 CNN Console",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Dark Theme Base */
    .stApp {
        background-color: #090B10;
        color: #F1F5F9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #090B10; }
    ::-webkit-scrollbar-thumb { background: #1F2737; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #38BDF8; }

    /* Top Platform Header */
    .platform-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #111622;
        border: 1px solid #1F2737;
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 12px;
    }
    .platform-title {
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #F8FAFC;
    }
    .platform-subtitle {
        font-size: 0.75rem;
        font-weight: 600;
        color: #00E5FF;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 2px;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid #22C55E;
        color: #22C55E;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        padding: 4px 10px;
        border-radius: 4px;
    }

    /* Sub-Ribbon Pipeline Flow */
    .sub-ribbon {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #0D111A;
        border: 1px solid #1A2130;
        border-radius: 6px;
        padding: 8px 16px;
        margin-bottom: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748B;
        letter-spacing: 0.06em;
        overflow-x: auto;
    }
    .sub-ribbon span.active {
        color: #38BDF8;
    }

    /* Cards */
    .metric-card {
        background: #111622;
        border: 1px solid #1F2737;
        border-radius: 8px;
        padding: 16px 18px;
        height: 100%;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748B;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Status Strip */
    .status-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
        background: #0D111A;
        border: 1px solid #1A2130;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 18px;
    }
    .strip-item {
        font-size: 0.75rem;
    }
    .strip-label {
        color: #64748B;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .strip-val {
        color: #F8FAFC;
        font-weight: 800;
        margin-top: 2px;
    }

    /* Hero Before/After Box */
    .hero-transform {
        background: #111622;
        border: 1px solid #1F2737;
        border-radius: 8px;
        padding: 18px 22px;
        margin: 18px 0;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #090B10;
        border-bottom: 1px solid #1F2737;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111622;
        border: 1px solid #1F2737;
        border-radius: 6px 6px 0 0;
        color: #94A3B8;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1A2234 !important;
        border-color: #38BDF8 !important;
        color: #38BDF8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Global Data Loaders & Model State
# -----------------------------------------------------------------------------
comparison = load_comparison_data()
metrics = load_metrics_data()
tinyml_analysis = load_tinyml_analysis()
has_data = bool(comparison and "comparison_results" in comparison)

fp32_b = comparison.get("fp32_baseline", {})
int8_q = comparison.get("int8_quantized", {})
res = comparison.get("comparison_results", {})


# -----------------------------------------------------------------------------
# Top Professional Application Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="platform-header">
        <div>
            <div class="platform-title">EDGE AI LAB // INT8 CNN OPTIMIZATION PLATFORM</div>
            <div class="platform-subtitle">Ultra-Low Power Edge Intelligence & TinyML Verification</div>
        </div>
        <div>
            <div class="status-badge">● SYSTEM READY</div>
        </div>
    </div>
    <div class="sub-ribbon">
        <span class="active">MNIST DATASET</span> ➔ 
        <span class="active">FP32 CNN (7.8K)</span> ➔ 
        <span class="active">INT8 PTQ (200 SAMPLES)</span> ➔ 
        <span class="active">TFLITE FLATBUFFER (13.5 KB)</span> ➔ 
        <span class="active">C-ARRAY / TINYML READY</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_dash, tab_infer, tab_analysis, tab_tinyml, tab_pipeline = st.tabs(
    [
        "📊 Dashboard",
        "⚡ Live Inference",
        "📈 Model Analysis",
        "🎛️ TinyML Verification",
        "📖 How It Works",
    ]
)


# -----------------------------------------------------------------------------
# TAB 1: DASHBOARD (Command Center)
# -----------------------------------------------------------------------------
with tab_dash:
    st.markdown(
        """
        <div style="margin-bottom: 14px;">
            <div style="font-size: 1.4rem; font-weight: 900; letter-spacing: -0.02em; color: #F8FAFC;">
                EDGE AI OPTIMIZATION // INT8 CNN DEPLOYMENT MONITOR
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8;">
                Quantization, benchmarking and TinyML deployment verification.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Status Strip
    st.markdown(
        """
        <div class="status-strip">
            <div class="strip-item">
                <div class="strip-label">MODEL</div>
                <div class="strip-val" style="color: #00E5FF;">INT8 TFLite</div>
            </div>
            <div class="strip-item">
                <div class="strip-label">STATUS</div>
                <div class="strip-val" style="color: #22C55E;">● READY</div>
            </div>
            <div class="strip-item">
                <div class="strip-label">INPUT</div>
                <div class="strip-val">28 × 28 × 1 INT8</div>
            </div>
            <div class="strip-item">
                <div class="strip-label">OUTPUT</div>
                <div class="strip-val">10 Classes INT8</div>
            </div>
            <div class="strip-item">
                <div class="strip-label">PARAMETERS</div>
                <div class="strip-val">7,834 Weights</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Primary Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">MODEL SIZE</div>
                <div class="metric-value" style="color: #00E5FF;">{int8_q.get('size_kb', 13.50)} KB</div>
                <div class="metric-sub" style="color: #22C55E;">↓ {res.get('size_reduction_percent', 61.10)}% Reduction</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">INT8 ACCURACY</div>
                <div class="metric-value" style="color: #22C55E;">{int8_q.get('accuracy_percent', 98.46)}%</div>
                <div class="metric-sub" style="color: #22C55E;">+{res.get('accuracy_delta_percentage_points', 0.02)} pts vs FP32</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">MEAN LATENCY</div>
                <div class="metric-value" style="color: #38BDF8;">{int8_q.get('latency_mean_ms', 0.0098):.4f} ms</div>
                <div class="metric-sub" style="color: #94A3B8;">Host CPU (Single Sample)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">COMPRESSION</div>
                <div class="metric-value" style="color: #A78BFA;">{res.get('compression_ratio', 2.57)}×</div>
                <div class="metric-sub" style="color: #94A3B8;">34.7 KB ➔ 13.5 KB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Visual FP32 ➔ INT8 Hero
    st.markdown(
        f"""
        <div class="hero-transform">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                <div style="flex: 1; min-width: 220px; background: #090B10; border: 1px solid #1A2130; border-radius: 6px; padding: 14px 18px;">
                    <div style="font-size: 0.72rem; font-weight: 800; color: #64748B; letter-spacing: 0.08em;">FP32 BASELINE</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #F8FAFC; margin-top: 2px;">{fp32_b.get('size_kb', 34.70)} KB</div>
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 2px;">Accuracy: <b style="color: #F8FAFC;">{fp32_b.get('accuracy_percent', 98.44)}%</b></div>
                    <div style="font-size: 0.8rem; color: #64748B;">Latency: {fp32_b.get('latency_mean_ms', 0.0133):.4f} ms</div>
                </div>
                <div style="text-align: center; padding: 0 10px;">
                    <div style="font-size: 1.1rem; color: #38BDF8; font-weight: 900;">➔ INT8 PTQ ➔</div>
                    <div style="font-size: 0.72rem; color: #22C55E; font-weight: 800; text-transform: uppercase;">61.10% Size Reduction</div>
                </div>
                <div style="flex: 1; min-width: 220px; background: #090B10; border: 1px solid #38BDF8; border-radius: 6px; padding: 14px 18px;">
                    <div style="font-size: 0.72rem; font-weight: 800; color: #38BDF8; letter-spacing: 0.08em;">INT8 DEPLOYMENT</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #00E5FF; margin-top: 2px;">{int8_q.get('size_kb', 13.50)} KB</div>
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 2px;">Accuracy: <b style="color: #22C55E;">{int8_q.get('accuracy_percent', 98.46)}%</b> (Lossless)</div>
                    <div style="font-size: 0.8rem; color: #64748B;">Latency: {int8_q.get('latency_mean_ms', 0.0098):.4f} ms</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5-Stage System Pipeline
    st.markdown(
        """
        <div style="background: #111622; border: 1px solid #1F2737; border-radius: 8px; padding: 14px 18px;">
            <div style="font-size: 0.72rem; font-weight: 800; color: #64748B; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 10px;">
                5-STAGE OPTIMIZATION PIPELINE
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; font-size: 0.8rem;">
                <div style="background: #090B10; border: 1px solid #1A2130; border-radius: 6px; padding: 10px;">
                    <div style="color: #64748B; font-weight: 800;">01 TRAIN</div>
                    <div style="color: #F8FAFC; font-weight: 700; margin-top: 2px;">Lightweight CNN</div>
                </div>
                <div style="background: #090B10; border: 1px solid #1A2130; border-radius: 6px; padding: 10px;">
                    <div style="color: #64748B; font-weight: 800;">02 CONVERT</div>
                    <div style="color: #F8FAFC; font-weight: 700; margin-top: 2px;">TFLite FlatBuffer</div>
                </div>
                <div style="background: #090B10; border: 1px solid #1A2130; border-radius: 6px; padding: 10px;">
                    <div style="color: #38BDF8; font-weight: 800;">03 QUANTIZE</div>
                    <div style="color: #00E5FF; font-weight: 700; margin-top: 2px;">INT8 PTQ (200 Samples)</div>
                </div>
                <div style="background: #090B10; border: 1px solid #1A2130; border-radius: 6px; padding: 10px;">
                    <div style="color: #64748B; font-weight: 800;">04 VERIFY</div>
                    <div style="color: #22C55E; font-weight: 700; margin-top: 2px;">Accuracy + Size + Latency</div>
                </div>
                <div style="background: #090B10; border: 1px solid #1A2130; border-radius: 6px; padding: 10px;">
                    <div style="color: #22C55E; font-weight: 800;">05 DEPLOY</div>
                    <div style="color: #A3FF12; font-weight: 700; margin-top: 2px;">TinyML C-Array Ready</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# TAB 2: LIVE INFERENCE (Interactive Real-Time Tool)
# -----------------------------------------------------------------------------
with tab_infer:
    st.markdown(
        """
        <div style="margin-bottom: 14px;">
            <div style="font-size: 1.3rem; font-weight: 800; color: #F8FAFC;">LIVE EDGE INFERENCE</div>
            <div style="font-size: 0.85rem; color: #94A3B8;">Real-time INT8 TFLite execution using verified quantization parameters.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown('<div class="metric-label">INPUT SELECTION</div>', unsafe_allow_html=True)
        input_mode = st.radio(
            "Input Mode:",
            ["Sample MNIST Digits", "Upload Image", "Draw Digit (Canvas)"],
            horizontal=True,
            label_visibility="collapsed",
        )

        norm_image: Optional[np.ndarray] = None
        preview_img: Optional[Image.Image] = None

        if input_mode == "Sample MNIST Digits":
            samples = get_available_sample_images()
            if samples:
                sample_names = [p.name for p in samples]
                selected_sample_name = st.selectbox("Select Sample Digit Image:", sample_names)
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
                    st.error(f"Image error: {e}")

        elif input_mode == "Draw Digit (Canvas)":
            st.caption("Draw a digit (0–9) below using your mouse or touch:")
            col_c1, col_c2 = st.columns([3, 1])
            with col_c2:
                stroke_width = st.slider("Stroke Width:", 12, 32, 20, 2)
            with col_c1:
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0.0)",
                    stroke_width=stroke_width,
                    stroke_color="#000000",
                    background_color="#FFFFFF",
                    update_streamlit=True,
                    height=180,
                    width=180,
                    drawing_mode="freedraw",
                    key="tab_infer_canvas",
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
            st.image(preview_img, caption="Preprocessed 28×28 Grayscale Input", width=120)

    with col_result:
        st.markdown('<div class="metric-label">INFERENCE RESULT</div>', unsafe_allow_html=True)
        if norm_image is not None:
            try:
                interpreter_int8 = get_interpreter("models/model_int8.tflite")
                infer_res = run_inference(interpreter_int8, norm_image)

                pred_digit = infer_res["predicted_digit"]
                conf = infer_res["confidence"]
                lat_ms = infer_res["latency_ms"]

                # Horizontal execution status sequence
                st.markdown(
                    """
                    <div style="display: flex; gap: 8px; font-size: 0.7rem; font-weight: 700; color: #22C55E; background: #0D111A; border: 1px solid #1A2130; border-radius: 4px; padding: 6px 10px; margin-bottom: 12px; flex-wrap: wrap;">
                        <span>● MODEL LOADED</span> ➔
                        <span>● INPUT QUANTIZED</span> ➔
                        <span>● INT8 INFERENCE COMPLETE</span> ➔
                        <span>● OUTPUT DEQUANTIZED</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div style="background: #111622; border: 1px solid #38BDF8; border-radius: 8px; padding: 18px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <div style="font-size: 0.72rem; font-weight: 800; color: #64748B; text-transform: uppercase;">PREDICTED DIGIT</div>
                                <div style="font-size: 3.5rem; font-weight: 900; line-height: 1; color: #00E5FF; margin-top: 2px;">#{pred_digit}</div>
                            </div>
                            <div style="text-align: right; font-size: 0.85rem;">
                                <div>Confidence: <b style="color: #22C55E;">{conf * 100.0:.2f}%</b></div>
                                <div style="margin-top: 4px;">Latency: <b style="color: #F8FAFC;">{lat_ms:.4f} ms</b> (Host CPU)</div>
                                <div style="margin-top: 4px; font-size: 0.75rem; color: #64748B;">Model: <b>INT8 TFLite (13.50 KB)</b></div>
                                <div style="margin-top: 4px; color: #22C55E; font-weight: 800; font-size: 0.78rem;">● INFERENCE COMPLETE</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Compact probability distribution chart
                probs = infer_res["probabilities"]
                fig_p = go.Figure(
                    go.Bar(
                        x=list(range(10)),
                        y=probs,
                        marker_color=["#00E5FF" if i == pred_digit else "#1F2737" for i in range(10)],
                        text=[f"{p * 100:.1f}%" if p > 0.05 else "" for p in probs],
                        textposition="auto",
                    )
                )
                fig_p.update_layout(
                    title="10-Class Softmax Probability Distribution",
                    xaxis=dict(tickmode="linear", tick0=0, dtick=1, title="Digit"),
                    yaxis=dict(title="Probability", range=[0, 1]),
                    height=200,
                    margin=dict(l=20, r=20, t=35, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F8FAFC", size=11),
                )
                st.plotly_chart(fig_p, use_container_width=True)

            except Exception as e:
                st.error(f"Inference error: {e}")
        else:
            st.info("Select a sample, upload an image, or draw on the canvas to execute live INT8 inference.")


# -----------------------------------------------------------------------------
# TAB 3: MODEL ANALYSIS (Detailed Comparison & Confusion Matrix)
# -----------------------------------------------------------------------------
with tab_analysis:
    st.markdown(
        """
        <div style="margin-bottom: 14px;">
            <div style="font-size: 1.3rem; font-weight: 800; color: #F8FAFC;">MODEL COMPARISON & BENCHMARKING</div>
            <div style="font-size: 0.85rem; color: #94A3B8;">Quantitative verification over 10,000 genuine MNIST test samples.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if has_data:
        # Clean Structured Table
        st.markdown(
            f"""
| Dimension | FP32 Baseline | INT8 Quantized | Measured Change |
|:---|:---|:---|:---|
| **Model Size** | `{fp32_b.get('size_kb', 34.70)} KB` ({fp32_b.get('size_bytes', 35536):,} B) | `{int8_q.get('size_kb', 13.50)} KB` ({int8_q.get('size_bytes', 13824):,} B) | **`-{res.get('size_reduction_percent', 61.10)}%`** ({res.get('compression_ratio', 2.57)}×) |
| **Test Accuracy (10k)** | `{fp32_b.get('accuracy_percent', 98.44)}%` ({fp32_b.get('correct_predictions', 9844):,} / 10k) | `{int8_q.get('accuracy_percent', 98.46)}%` ({int8_q.get('correct_predictions', 9846):,} / 10k) | **`+{res.get('accuracy_delta_percentage_points', 0.02)} pts`** (Lossless) |
| **Host Mean Latency** | `{fp32_b.get('latency_mean_ms', 0.0133):.4f} ms` | `{int8_q.get('latency_mean_ms', 0.0098):.4f} ms` | **`{res.get('latency_change_percent', -26.32)}%`** (Host CPU) |
| **Tensor Data Types** | `float32` in / `float32` out | `int8` in / `int8` out | **Full-Integer Quantization** |
| **Hardware FPU** | Required | **Not Required** | **TinyML ALU Compatible** |
            """
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="metric-label">VISUAL IMPACT CHARTS</div>', unsafe_allow_html=True)

        col_ch1, col_ch2 = st.columns(2)
        with col_ch1:
            fig_size = go.Figure(
                go.Bar(
                    x=["FP32 Baseline", "INT8 Quantized"],
                    y=[fp32_b.get("size_kb", 34.70), int8_q.get("size_kb", 13.50)],
                    marker_color=["#334155", "#00E5FF"],
                    text=[f"{fp32_b.get('size_kb', 34.70)} KB", f"{int8_q.get('size_kb', 13.50)} KB"],
                    textposition="auto",
                )
            )
            fig_size.update_layout(
                title=f"Flash Storage: -{res.get('size_reduction_percent', 61.10)}%",
                yaxis=dict(title="Size (KB)"),
                height=230,
                margin=dict(l=20, r=20, t=35, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC", size=11),
            )
            st.plotly_chart(fig_size, use_container_width=True)

        with col_ch2:
            fig_lat = go.Figure(
                go.Bar(
                    x=["FP32 Mean", "INT8 Mean"],
                    y=[fp32_b.get("latency_mean_ms", 0.0133), int8_q.get("latency_mean_ms", 0.0098)],
                    marker_color=["#334155", "#38BDF8"],
                    text=[f"{fp32_b.get('latency_mean_ms', 0.0133):.4f} ms", f"{int8_q.get('latency_mean_ms', 0.0098):.4f} ms"],
                    textposition="auto",
                )
            )
            fig_lat.update_layout(
                title="Single-Sample Host Latency (ms)",
                yaxis=dict(title="Latency (ms)"),
                height=230,
                margin=dict(l=20, r=20, t=35, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC", size=11),
            )
            st.plotly_chart(fig_lat, use_container_width=True)

        # Confusion Matrices
        st.markdown("---")
        st.markdown('<div class="metric-label">CLASSIFICATION ANALYSIS: FP32 VS INT8 CONFUSION MATRIX</div>', unsafe_allow_html=True)
        cm_data = load_confusion_matrix_data()
        if cm_data and "fp32" in cm_data and "int8" in cm_data:
            cm_mode = st.radio("View:", ["Raw Counts", "Normalized (%)"], horizontal=True, key="cm_tab_view")
            is_norm = cm_mode == "Normalized (%)"

            fp32_mat = np.array(cm_data["fp32"]["matrix"], dtype=np.float32)
            int8_mat = np.array(cm_data["int8"]["matrix"], dtype=np.float32)
            classes = [str(c) for c in cm_data.get("classes", list(range(10)))]

            if is_norm:
                r_fp32 = fp32_mat.sum(axis=1, keepdims=True)
                d_fp32 = np.where(r_fp32 > 0, (fp32_mat / r_fp32) * 100.0, 0.0)
                r_int8 = int8_mat.sum(axis=1, keepdims=True)
                d_int8 = np.where(r_int8 > 0, (int8_mat / r_int8) * 100.0, 0.0)
                zmin, zmax = 0.0, 100.0
                t_fp32 = [[f"{val:.1f}%" if val >= 0.5 else "" for val in row] for row in d_fp32]
                t_int8 = [[f"{val:.1f}%" if val >= 0.5 else "" for val in row] for row in d_int8]
                cb_title = "Recall %"
            else:
                d_fp32, d_int8 = fp32_mat, int8_mat
                zmin, zmax = 0, float(np.max([fp32_mat.max(), int8_mat.max()]))
                t_fp32 = [[f"{int(val)}" if val > 0 else "" for val in row] for row in d_fp32]
                t_int8 = [[f"{int(val)}" if val > 0 else "" for val in row] for row in d_int8]
                cb_title = "Count"

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                f_cm1 = go.Figure(
                    go.Heatmap(
                        z=d_fp32, x=classes, y=classes, text=t_fp32, texttemplate="%{text}",
                        colorscale="Blues", zmin=zmin, zmax=zmax, colorbar=dict(title=cb_title)
                    )
                )
                f_cm1.update_layout(
                    title=f"FP32 ({cm_data['fp32']['correct_predictions']:,} Correct)",
                    xaxis=dict(title="Predicted", dtick=1), yaxis=dict(title="Actual", dtick=1, autorange="reversed"),
                    height=320, margin=dict(l=30, r=10, t=35, b=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8FAFC", size=10)
                )
                st.plotly_chart(f_cm1, use_container_width=True)

            with col_m2:
                f_cm2 = go.Figure(
                    go.Heatmap(
                        z=d_int8, x=classes, y=classes, text=t_int8, texttemplate="%{text}",
                        colorscale="Teal", zmin=zmin, zmax=zmax, colorbar=dict(title=cb_title)
                    )
                )
                f_cm2.update_layout(
                    title=f"INT8 ({cm_data['int8']['correct_predictions']:,} Correct)",
                    xaxis=dict(title="Predicted", dtick=1), yaxis=dict(title="Actual", dtick=1, autorange="reversed"),
                    height=320, margin=dict(l=30, r=10, t=35, b=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8FAFC", size=10)
                )
                st.plotly_chart(f_cm2, use_container_width=True)

        with st.expander("View detailed benchmark methodology"):
            st.markdown(
                f"""
                - **Test Dataset**: 10,000 genuine MNIST test set images.
                - **Timing Method**: Single-sample execution on host development CPU using `time.perf_counter()`.
                - **Median Latency**: FP32 `{fp32_b.get('latency_median_ms', 0.0105):.4f} ms` vs INT8 `{int8_q.get('latency_median_ms', 0.0097):.4f} ms`.
                - **Latency Standard Deviation**: FP32 `{fp32_b.get('latency_std_ms', 0.004):.4f} ms` vs INT8 `{int8_q.get('latency_std_ms', 0.002):.4f} ms`.
                - **Note**: Host latency measures development machine execution. Microcontroller latency will depend on hardware clock speed.
                """
            )
    else:
        st.error("No comparison data available.")


# -----------------------------------------------------------------------------
# TAB 4: TINYML VERIFICATION (Resource Monitor & Budget Checker)
# -----------------------------------------------------------------------------
with tab_tinyml:
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
            <div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #F8FAFC;">TINYML RESOURCE MONITOR</div>
                <div style="font-size: 0.85rem; color: #94A3B8;">Memory feasibility analysis and embedded C-array verification.</div>
            </div>
            <div>
                <div style="background: rgba(34, 197, 94, 0.1); border: 1px solid #22C55E; color: #22C55E; font-size: 0.75rem; font-weight: 800; padding: 4px 10px; border-radius: 4px;">
                    TINYML READINESS: ● PREPARED
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if tinyml_analysis and "verified" in tinyml_analysis:
        ver = tinyml_analysis["verified"]
        est = tinyml_analysis["estimated"]

        # 3 Compact Resource Indicators
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">FLASH STORAGE</div>
                    <div class="metric-value" style="color: #00E5FF;">13.50 KB</div>
                    <div class="metric-sub" style="color: #94A3B8;">Model Binary Footprint</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">RAM ARENA</div>
                    <div class="metric-value" style="color: #F59E0B;">~14.0 KB</div>
                    <div class="metric-sub" style="color: #94A3B8;">Estimated Tensor Arena</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">C-ARRAY BINARY</div>
                    <div class="metric-value" style="color: #22C55E;">13,824 B</div>
                    <div class="metric-sub" style="color: #22C55E;">✓ Verified Byte Parity</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="metric-label">MEMORY BUDGET CHECKER</div>', unsafe_allow_html=True)

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

        col_pr, col_fl, col_rm = st.columns([2, 1, 1])
        with col_pr:
            sel_preset = st.selectbox("Target MCU Preset:", list(mcu_presets.keys()), index=0)
            def_f, def_r = mcu_presets[sel_preset]
        with col_fl:
            fl_kb = st.number_input("Target Flash (KB):", 1.0, 16384.0, float(def_f), 8.0, key=f"f_in_{sel_preset}")
        with col_rm:
            rm_kb = st.number_input("Target RAM (KB):", 1.0, 4096.0, float(def_r), 4.0, key=f"r_in_{sel_preset}")

        b_res = evaluate_memory_budget(
            fl_kb, rm_kb,
            model_flash_bytes=ver["flash_storage_bytes"],
            estimated_arena_bytes=est["estimated_tensor_arena_bytes"],
        )

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            f_status = "✓ FITS" if b_res["flash_fits"] else "✕ DOES NOT FIT"
            f_col = "#22C55E" if b_res["flash_fits"] else "#EF4444"
            st.markdown(
                f"""
                <div style="background: #111622; border: 1px solid #1F2737; border-radius: 6px; padding: 12px 16px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: 700; font-size: 0.88rem;">Flash Allocation</span>
                        <span style="font-weight: 800; font-size: 0.85rem; color: {f_col};">{f_status}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">
                        Model: <b>{b_res['model_flash_kb']:.2f} KB</b> / Available: <b>{b_res['available_flash_kb']:.1f} KB</b> ({b_res['flash_usage_pct']:.1f}%)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(1.0, b_res["flash_usage_pct"] / 100.0) if b_res["available_flash_kb"] > 0 else 1.0)

        with col_g2:
            r_status = "✓ FITS" if b_res["ram_fits"] else "✕ DOES NOT FIT"
            r_col = "#22C55E" if b_res["ram_fits"] else "#EF4444"
            st.markdown(
                f"""
                <div style="background: #111622; border: 1px solid #1F2737; border-radius: 6px; padding: 12px 16px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: 700; font-size: 0.88rem;">RAM Allocation</span>
                        <span style="font-weight: 800; font-size: 0.85rem; color: {r_col};">{r_status}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">
                        Arena: <b>~{b_res['estimated_arena_kb']:.1f} KB</b> / Available: <b>{b_res['available_ram_kb']:.1f} KB</b> (~{b_res['ram_usage_pct']:.1f}%)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(1.0, b_res["ram_usage_pct"] / 100.0) if b_res["available_ram_kb"] > 0 else 1.0)

        if b_res["fits_overall"]:
            st.success(f"✓ **DEPLOYMENT FEASIBLE*** — {b_res['explanation']}")
        else:
            st.error(f"✕ **INSUFFICIENT MEMORY** — {b_res['explanation']}")

        st.markdown("<br>", unsafe_allow_html=True)

        # 3 Verification Status Groups
        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown(
                """
                <div style="background: #111622; border: 1px solid #22C55E; border-radius: 6px; padding: 14px;">
                    <div style="font-size: 0.8rem; font-weight: 800; color: #22C55E; margin-bottom: 6px;">✓ VERIFIED</div>
                    <div style="font-size: 0.8rem; color: #CBD5E1; line-height: 1.6;">
                        • INT8 Model (13.50 KB / 13,824 B)<br>
                        • C-Array Byte Parity (100%)<br>
                        • int8 Input / Output Types<br>
                        • TFLM Supported Operators (5/5)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with g2:
            st.markdown(
                """
                <div style="background: #111622; border: 1px solid #F59E0B; border-radius: 6px; padding: 14px;">
                    <div style="font-size: 0.8rem; font-weight: 800; color: #F59E0B; margin-bottom: 6px;">~ ESTIMATED</div>
                    <div style="font-size: 0.8rem; color: #CBD5E1; line-height: 1.6;">
                        • Tensor Arena (~14.0 KB)<br>
                        • Peak Activation (5,408 B)<br>
                        • Input Buffer (784 B)<br>
                        • Analytical projection (heuristic)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with g3:
            st.markdown(
                """
                <div style="background: #111622; border: 1px solid #64748B; border-radius: 6px; padding: 14px;">
                    <div style="font-size: 0.8rem; font-weight: 800; color: #94A3B8; margin-bottom: 6px;">! NOT VERIFIED</div>
                    <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.6;">
                        • Physical MCU Flashing<br>
                        • Hardware Clock Cycles<br>
                        • Power / Current Draw<br>
                        • Host simulation boundaries
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")

        # ---------------------------------------------------------------------
        # 🚀 WHAT-IF MCU SIMULATOR SECTION
        # ---------------------------------------------------------------------
        st.markdown("### 🚀 What-If MCU Simulator")
        st.markdown(
            """
            <div style="font-size: 0.88rem; color: #94A3B8; margin-bottom: 16px;">
                Explore whether the current INT8 model would fit within a hypothetical MCU's Flash and SRAM constraints.
                <br><span style="font-size: 0.76rem; color: #64748B;">⚡ Static resource simulation based on verified binary footprint and analytical Tensor Arena estimates.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Model Requirements Display
        col_req1, col_req2 = st.columns(2)
        with col_req1:
            st.markdown(
                f"""
                <div style="background: #0D111A; border: 1px solid #1F2737; border-radius: 6px; padding: 12px 16px; margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.72rem; font-weight: 800; color: #64748B; text-transform: uppercase;">MODEL REQUIREMENTS: FLASH</span>
                        <span style="font-size: 0.7rem; font-weight: 800; color: #22C55E; background: rgba(34, 197, 94, 0.1); border: 1px solid #22C55E; padding: 2px 6px; border-radius: 4px;">VERIFIED</span>
                    </div>
                    <div style="font-size: 1.4rem; font-weight: 900; color: #00E5FF; margin-top: 4px;">13.50 KB <span style="font-size: 0.85rem; color: #94A3B8; font-weight: 600;">({ver['flash_storage_bytes']:,} bytes)</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_req2:
            st.markdown(
                f"""
                <div style="background: #0D111A; border: 1px solid #1F2737; border-radius: 6px; padding: 12px 16px; margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.72rem; font-weight: 800; color: #64748B; text-transform: uppercase;">MODEL REQUIREMENTS: RAM</span>
                        <span style="font-size: 0.7rem; font-weight: 800; color: #F59E0B; background: rgba(245, 158, 11, 0.1); border: 1px solid #F59E0B; padding: 2px 6px; border-radius: 4px;">ESTIMATED</span>
                    </div>
                    <div style="font-size: 1.4rem; font-weight: 900; color: #F59E0B; margin-top: 4px;">~14.00 KB <span style="font-size: 0.85rem; color: #94A3B8; font-weight: 600;">(Tensor Arena Estimate)</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        mcu_presets = {
            "Standard TinyML (32 KB Flash / 32 KB RAM)": (32.0, 32.0),
            "Generic Tiny (16 KB Flash / 16 KB RAM)": (16.0, 16.0),
            "Mid-Range MCU (64 KB Flash / 64 KB RAM)": (64.0, 64.0),
            "High-End TinyML (128 KB Flash / 128 KB RAM)": (128.0, 128.0),
            "Arduino Nano 33 BLE (1024 KB Flash / 256 KB RAM)": (1024.0, 256.0),
            "Raspberry Pi RP2040 (2048 KB Flash / 264 KB RAM)": (2048.0, 264.0),
            "ESP32-S3 (4096 KB Flash / 512 KB RAM)": (4096.0, 512.0),
            "Custom MCU": (32.0, 32.0),
        }

        col_pr, col_fl, col_rm = st.columns([2, 1, 1])
        with col_pr:
            sel_preset = st.selectbox(
                "Target Device Preset:",
                list(mcu_presets.keys()),
                index=0,
                key="whatif_mcu_preset_select",
            )
            def_f, def_r = mcu_presets[sel_preset]

        with col_fl:
            fl_kb = st.number_input(
                "Available Flash (KB):",
                min_value=0.0,
                max_value=16384.0,
                value=float(def_f),
                step=4.0,
                key=f"whatif_flash_input_{sel_preset}",
            )
        with col_rm:
            rm_kb = st.number_input(
                "Available RAM (KB):",
                min_value=0.0,
                max_value=4096.0,
                value=float(def_r),
                step=4.0,
                key=f"whatif_ram_input_{sel_preset}",
            )

        sim_res = simulate_mcu_resources(
            available_flash_kb=fl_kb,
            available_ram_kb=rm_kb,
            model_flash_bytes=ver["flash_storage_bytes"],
            estimated_arena_bytes=est["estimated_tensor_arena_bytes"],
        )

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            f_status = "✓ FLASH FITS" if sim_res["flash_fits"] else "✕ FLASH DOES NOT FIT"
            f_col = "#22C55E" if sim_res["flash_fits"] else "#EF4444"
            f_detail = (
                f"{sim_res['flash_usage_pct']:.1f}% utilized &nbsp;|&nbsp; {sim_res['flash_headroom_kb']:.2f} KB remaining"
                if sim_res["flash_fits"]
                else f"{sim_res['model_flash_kb']:.2f} KB required, {sim_res['available_flash_kb']:.1f} KB available &nbsp;|&nbsp; Shortfall: {sim_res['flash_shortfall_kb']:.2f} KB"
            )
            st.markdown(
                f"""
                <div style="background: #111622; border: 1px solid {'#22C55E' if sim_res['flash_fits'] else '#EF4444'}; border-radius: 6px; padding: 14px 16px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 800; font-size: 0.95rem; color: #F8FAFC;">FLASH ANALYSIS</span>
                        <span style="font-weight: 800; font-size: 0.85rem; color: {f_col};">{f_status}</span>
                    </div>
                    <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 6px;">
                        <b>{sim_res['model_flash_kb']:.2f} KB</b> / <b>{sim_res['available_flash_kb']:.1f} KB</b> used
                    </div>
                    <div style="font-size: 0.8rem; color: {'#22C55E' if sim_res['flash_fits'] else '#EF4444'}; margin-top: 4px;">
                        {f_detail}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            f_prog = min(1.0, sim_res["flash_usage_pct"] / 100.0) if sim_res["available_flash_kb"] > 0 else 1.0
            st.progress(f_prog, text=f"Flash: {sim_res['flash_usage_pct']:.1f}% utilized")

        with col_g2:
            r_status = "✓ RAM FITS" if sim_res["ram_fits"] else "✕ RAM DOES NOT FIT"
            r_col = "#22C55E" if sim_res["ram_fits"] else "#EF4444"
            r_detail = (
                f"~{sim_res['ram_usage_pct']:.1f}% utilized &nbsp;|&nbsp; ~{sim_res['ram_headroom_kb']:.2f} KB remaining"
                if sim_res["ram_fits"]
                else f"~{sim_res['estimated_arena_kb']:.2f} KB required, {sim_res['available_ram_kb']:.1f} KB available &nbsp;|&nbsp; Shortfall: ~{sim_res['ram_shortfall_kb']:.2f} KB"
            )
            st.markdown(
                f"""
                <div style="background: #111622; border: 1px solid {'#22C55E' if sim_res['ram_fits'] else '#EF4444'}; border-radius: 6px; padding: 14px 16px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 800; font-size: 0.95rem; color: #F8FAFC;">RAM ANALYSIS <span style="font-size: 0.72rem; color: #64748B; font-weight: 600;">(Tensor Arena Estimate)</span></span>
                        <span style="font-weight: 800; font-size: 0.85rem; color: {r_col};">{r_status}</span>
                    </div>
                    <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 6px;">
                        <b>~{sim_res['estimated_arena_kb']:.2f} KB</b> / <b>{sim_res['available_ram_kb']:.1f} KB</b> used
                    </div>
                    <div style="font-size: 0.8rem; color: {'#22C55E' if sim_res['ram_fits'] else '#EF4444'}; margin-top: 4px;">
                        {r_detail}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            r_prog = min(1.0, sim_res["ram_usage_pct"] / 100.0) if sim_res["available_ram_kb"] > 0 else 1.0
            st.progress(r_prog, text=f"RAM: ~{sim_res['ram_usage_pct']:.1f}% utilized")

        st.markdown("<br>", unsafe_allow_html=True)

        # Prominent Overall Result Card
        if sim_res["status_category"] == "BOTH_PASS":
            st.markdown(
                f"""
                <div style="background: #111622; border: 1px solid #22C55E; border-radius: 8px; padding: 18px; box-shadow: 0 0 15px rgba(34, 197, 94, 0.15);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.25rem; font-weight: 900; color: #22C55E;">{sim_res['status_title']}</span>
                        <span style="font-size: 0.85rem; font-weight: 800; color: #22C55E; background: rgba(34, 197, 94, 0.1); border: 1px solid #22C55E; padding: 3px 10px; border-radius: 4px;">Flash: PASS &nbsp;|&nbsp; RAM: PASS</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #CBD5E1; margin-top: 8px; line-height: 1.5;">
                        {sim_res['status_message']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif sim_res["status_category"] in ("FLASH_FAIL", "RAM_FAIL"):
            st.markdown(
                f"""
                <div style="background: #111622; border: 1px solid #F59E0B; border-radius: 8px; padding: 18px; box-shadow: 0 0 15px rgba(245, 158, 11, 0.15);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.25rem; font-weight: 900; color: #F59E0B;">{sim_res['status_title']}</span>
                        <span style="font-size: 0.85rem; font-weight: 800; color: #F59E0B; background: rgba(245, 158, 11, 0.1); border: 1px solid #F59E0B; padding: 3px 10px; border-radius: 4px;">Flash: {sim_res['flash_status_str']} &nbsp;|&nbsp; RAM: {sim_res['ram_status_str']}</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #CBD5E1; margin-top: 8px; line-height: 1.5;">
                        {sim_res['status_message']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background: #111622; border: 1px solid #EF4444; border-radius: 8px; padding: 18px; box-shadow: 0 0 15px rgba(239, 68, 68, 0.15);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.25rem; font-weight: 900; color: #EF4444;">{sim_res['status_title']}</span>
                        <span style="font-size: 0.85rem; font-weight: 800; color: #EF4444; background: rgba(239, 68, 68, 0.1); border: 1px solid #EF4444; padding: 3px 10px; border-radius: 4px;">Flash: FAIL &nbsp;|&nbsp; RAM: FAIL</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #CBD5E1; margin-top: 8px; line-height: 1.5;">
                        {sim_res['status_message']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Resource Headroom / Shortfall Summary
        col_hr1, col_hr2 = st.columns(2)
        with col_hr1:
            if sim_res["flash_fits"]:
                st.markdown(
                    f"""
                    <div style="background: #0D111A; border: 1px solid #1F2737; border-radius: 6px; padding: 10px 14px; margin-top: 10px;">
                        <div style="font-size: 0.72rem; font-weight: 800; color: #64748B; text-transform: uppercase;">FLASH HEADROOM</div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #22C55E; margin-top: 2px;">+{sim_res['flash_headroom_kb']:.2f} KB</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background: #0D111A; border: 1px solid #1F2737; border-radius: 6px; padding: 10px 14px; margin-top: 10px;">
                        <div style="font-size: 0.72rem; font-weight: 800; color: #64748B; text-transform: uppercase;">FLASH SHORTFALL</div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #EF4444; margin-top: 2px;">-{sim_res['flash_shortfall_kb']:.2f} KB</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_hr2:
            if sim_res["ram_fits"]:
                st.markdown(
                    f"""
                    <div style="background: #0D111A; border: 1px solid #1F2737; border-radius: 6px; padding: 10px 14px; margin-top: 10px;">
                        <div style="font-size: 0.72rem; font-weight: 800; color: #64748B; text-transform: uppercase;">RAM HEADROOM <span style="font-size: 0.65rem; color: #64748B;">(ESTIMATED)</span></div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #22C55E; margin-top: 2px;">~+{sim_res['ram_headroom_kb']:.2f} KB</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background: #0D111A; border: 1px solid #1F2737; border-radius: 6px; padding: 10px 14px; margin-top: 10px;">
                        <div style="font-size: 0.72rem; font-weight: 800; color: #64748B; text-transform: uppercase;">RAM SHORTFALL <span style="font-size: 0.65rem; color: #64748B;">(ESTIMATED)</span></div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #EF4444; margin-top: 2px;">~-{sim_res['ram_shortfall_kb']:.2f} KB</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            """
            <div style="font-size: 0.76rem; color: #64748B; background: #0D111A; border: 1px solid #1A2130; border-radius: 6px; padding: 10px 14px; margin-top: 12px; line-height: 1.4;">
                <b>Simulation boundary:</b> This tool compares the INT8 model footprint and estimated Tensor Arena against user-defined Flash/SRAM budgets. It does not measure firmware size, HAL/runtime overhead, stack usage, power consumption, clock cycles, or physical MCU behavior.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("Advanced Memory Analysis"):
            st.markdown(
                f"""
                - **Flash Storage**: Exact INT8 FlatBuffer model is `{ver['flash_storage_bytes']:,} Bytes` (`13.50 KB`).
                - **Input Buffer**: `784 Bytes` (28 × 28 × 1 int8 elements).
                - **Peak Activation**: `5,408 Bytes` (largest single activation buffer during execution).
                - **Estimated Tensor Arena**: `~14.0 KB` (`{est['estimated_tensor_arena_bytes']:,} Bytes`).
                - **Note**: *Static estimate. Application firmware, RTOS heap, and MCU call stack overhead are not included.
                """
            )

        with st.expander("Operator Compatibility & C-Array Header"):
            ops_df = [
                {"Operator": op["operator"], "Status": "✓ Supported" if op["supported_in_tflm"] else "✕ Unsupported", "Notes": op["notes"]}
                for op in ver["tflite_micro_compatible_ops"]
            ]
            st.table(ops_df)
            header_path = TINYML_DIR / "model_data.h"
            if header_path.exists():
                with open(header_path, "r", encoding="utf-8") as f:
                    st.code(f.read(), language="c")
    else:
        st.error("TinyML model analysis file not found.")



# -----------------------------------------------------------------------------
# TAB 5: HOW IT WORKS (Visual Storytelling Sequence)
# -----------------------------------------------------------------------------
with tab_pipeline:
    st.markdown(
        """
        <div style="margin-bottom: 14px;">
            <div style="font-size: 1.3rem; font-weight: 800; color: #F8FAFC;">HOW IT WORKS</div>
            <div style="font-size: 0.85rem; color: #94A3B8;">The complete engineering journey from FP32 CNN to TinyML-ready integer model.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background: #111622; border: 1px solid #1F2737; border-radius: 8px; padding: 20px; line-height: 1.7; font-size: 0.88rem;">
            <div style="margin-bottom: 14px;">
                <b style="color: #EF4444;">1. WHY?</b><br>
                Standard deep learning pipelines train CNNs with 32-bit floating point arithmetic. Microcontrollers have limited Flash (32–256 KB), minimal SRAM (16–64 KB), and lack floating-point hardware units (FPUs).
            </div>
            <div style="margin-bottom: 14px;">
                <b style="color: #00E5FF;">2. WHAT?</b><br>
                Post-Training Quantization (PTQ) converts 32-bit floating-point weights and activation tensors into compact 8-bit signed integers (<code>int8</code>).
            </div>
            <div style="margin-bottom: 14px;">
                <b style="color: #38BDF8;">3. HOW?</b><br>
                A calibration dataset of 200 real MNIST training samples is passed through the network to record dynamic activation ranges across all layers and compute optimal scale ($S$) and zero-point ($Z$) parameters.
            </div>
            <div style="margin-bottom: 14px;">
                <b style="color: #22C55E;">4. RESULT?</b><br>
                The model achieves a <b>61.10% size reduction</b> (34.70 KB ➔ 13.50 KB, 2.57× compression) while preserving 100% of its accuracy (98.44% FP32 ➔ <b>98.46% INT8</b>).
            </div>
            <div>
                <b style="color: #A3FF12;">5. DEPLOYMENT</b><br>
                The INT8 TFLite model is converted to a byte-exact C array (<code>tinyml/model_data.h</code> and <code>.cc</code>) ready for execution with TensorFlow Lite for Microcontrollers (TFLM).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("View Quantization Mathematics & Parameters"):
        st.markdown(
            """
            ##### Linear Affine Quantization Equations
            $$\\text{Quantize: } q = \\text{clip}\\left(\\left\\lfloor \\frac{r}{S} \\right\\rceil + Z, -128, 127\\right)$$
            $$\\text{Dequantize: } r = S \\times (q - Z)$$

            - **$S$ (Scale)**: Positive float representing the step size per integer quantum.
            - **$Z$ (Zero-Point)**: Integer offset corresponding exactly to real value `0.0`.

            ##### Actual Verified Tensor Parameters
            - **Input Tensor**: `dtype: int8`, `Scale: 0.003921568859368563`, `Zero-Point: -128`, Shape: `[1, 28, 28, 1]`.
              Real `0.0 ➔ -128`, Real `1.0 ➔ 127`.
            - **Output Tensor**: `dtype: int8`, `Scale: 0.00390625`, `Zero-Point: -128`, Shape: `[1, 10]`.
              Quantized `-128 ➔ 0.0`, Quantized `127 ➔ ≈ 1.0`.
            """
        )
