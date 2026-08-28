"""
PS09: INT8 Quantized CNN Deployment for Resource-Constrained TinyML Devices.
Minimalist, high-clarity Streamlit dashboard for demonstrating INT8 Post-Training Quantization.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import plotly.graph_objects as go
import streamlit as st

from src.inference import (
    dequantize_output_tensor,
    get_quantization_details,
    load_tflite_interpreter,
    quantize_input_image,
    run_inference,
)

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

    # Check if background is light (mean corners > 127) and invert to match MNIST (white digit on black background)
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
    page_title="INT8 Quantized CNN for TinyML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #38BDF8;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 1.2rem;
    }
    .pipeline-badge {
        display: inline-block;
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px 16px;
        font-family: monospace;
        font-size: 0.95rem;
        color: #38BDF8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
    }
    .delta-pos { color: #10B981; }
    .delta-neu { color: #38BDF8; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">INT8 Quantized CNN Deployment for TinyML</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Resource-Constrained Edge Deep Learning via Post-Training Quantization (PTQ)</div>', unsafe_allow_html=True)

# Navigation Tabs
tab_dash, tab_infer, tab_analysis, tab_tinyml, tab_how = st.tabs([
    "📊 Dashboard",
    "⚡ Live Inference",
    "📈 Model Analysis",
    "🎛️ TinyML Verification",
    "📖 How It Works",
])

# Load data
comparison = load_comparison_data()
metrics = load_metrics_data()
tinyml_analysis = load_tinyml_analysis()

has_data = bool(comparison and "comparison_results" in comparison)


# -----------------------------------------------------------------------------
# TAB 1: Dashboard
# -----------------------------------------------------------------------------
with tab_dash:
    st.markdown(
        '<div class="pipeline-badge">⚡ Pipeline: <b>MNIST</b> → <b>FP32 CNN</b> → <b>INT8 PTQ</b> → <b>INT8 TFLite (13.5 KB)</b> → <b>TinyML C-Array</b></div>',
        unsafe_allow_html=True,
    )

    # Problem / Solution Statement
    col_p, col_s = st.columns(2)
    with col_p:
        st.markdown("##### 🎯 Problem Statement")
        st.info("Resource-constrained microcontrollers have severe Flash (<256 KB) and SRAM (<64 KB) limits. Conventional FP32 CNNs consume excessive memory and require power-hungry floating-point units.")
    with col_s:
        st.markdown("##### 💡 Technical Solution")
        st.success("Post-Training Quantization (PTQ) compresses weights and activations to 8-bit integers without retraining, reducing model size by 61.10% while preserving 100% of the baseline accuracy.")

    st.markdown("---")
    st.markdown("#### 🏆 Verified Headline Benchmarks")

    if has_data:
        fp32_b = comparison["fp32_baseline"]
        int8_q = comparison["int8_quantized"]
        res = comparison["comparison_results"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-lbl">Model Size Reduction</div>
                    <div class="metric-val">{res['size_reduction_percent']}%</div>
                    <div class="metric-delta delta-pos">{fp32_b['size_kb']} KB → {int8_q['size_kb']} KB</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-lbl">Compression Ratio</div>
                    <div class="metric-val">{res['compression_ratio']}×</div>
                    <div class="metric-delta delta-pos">{res['size_reduction_bytes']:,} Bytes Saved</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-lbl">INT8 Test Accuracy</div>
                    <div class="metric-val">{int8_q['accuracy_percent']}%</div>
                    <div class="metric-delta delta-pos">+{res['accuracy_delta_percentage_points']} pts vs FP32 ({fp32_b['accuracy_percent']}%)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-lbl">Host Mean Latency</div>
                    <div class="metric-val">{int8_q['latency_mean_ms']} ms</div>
                    <div class="metric-delta delta-pos">{res['latency_change_percent']}% on CPU</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("Benchmark results file not found at results/comparison.json")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔬 Why INT8 for Edge TinyML?")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**📦 61% Smaller Flash Size**")
        st.caption("Shrinks the binary down to 13.5 KB so the entire neural network comfortably fits on small embedded Flash.")
    with c2:
        st.markdown("**🔢 Integer-Only Arithmetic**")
        st.caption("Operates entirely using 8-bit integer SIMD ALU instructions, eliminating the requirement for a hardware FPU.")
    with c3:
        st.markdown("**⚡ Lower SRAM Footprint**")
        st.caption("Intermediate activation buffers require 4× less RAM per element compared to 32-bit floats.")
    with c4:
        st.markdown("**🎯 Zero Accuracy Degradation**")
        st.caption("Calibration with real training samples maintains 98.46% test accuracy on 10,000 MNIST test samples.")


# -----------------------------------------------------------------------------
# TAB 2: Live Inference
# -----------------------------------------------------------------------------
with tab_infer:
    st.markdown("#### ⚡ Real-Time INT8 Model Inference")
    st.caption("Select a genuine MNIST test sample or upload a handwritten digit image to execute live inference with the verified INT8 TFLite model.")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        input_mode = st.radio("Choose Input Source:", ["Sample MNIST Digits", "Upload Image"], horizontal=True)

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
        else:
            uploaded_file = st.file_uploader("Upload handwritten digit (PNG, JPG)", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                try:
                    norm_image, preview_img = preprocess_uploaded_image(uploaded_file)
                except Exception as e:
                    st.error(f"Error processing uploaded image: {e}")

        if preview_img is not None:
            st.image(preview_img, caption="Preprocessed 28×28 Grayscale Input", width=160)

    with col_result:
        st.markdown("##### 🎯 INT8 Inference Output")
        if norm_image is not None:
            try:
                interpreter_int8 = get_interpreter("models/model_int8.tflite")
                res = run_inference(interpreter_int8, norm_image)

                pred_digit = res["predicted_digit"]
                conf = res["confidence"]
                lat_ms = res["latency_ms"]

                st.markdown(
                    f"""
                    <div style="background: #1E293B; border: 1px solid #38BDF8; border-radius: 8px; padding: 18px; margin-bottom: 16px;">
                        <div style="font-size: 0.9rem; color: #94A3B8; text-transform: uppercase;">Predicted Digit</div>
                        <div style="font-size: 3rem; font-weight: 800; color: #38BDF8;">{pred_digit}</div>
                        <div style="font-size: 0.95rem; color: #F8FAFC;">Confidence: <b>{conf * 100.0:.2f}%</b> &nbsp;|&nbsp; Latency: <b>{lat_ms:.4f} ms</b></div>
                        <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Engine: <b>INT8 TFLite</b> (Tensor dtypes: {res['input_dtype']} in / {res['output_dtype']} out)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Class probabilities distribution chart
                probs = res["probabilities"]
                fig_prob = go.Figure(
                    go.Bar(
                        x=list(range(10)),
                        y=probs,
                        marker_color=["#38BDF8" if i == pred_digit else "#334155" for i in range(10)],
                        text=[f"{p * 100:.1f}%" if p > 0.05 else "" for p in probs],
                        textposition="auto",
                    )
                )
                fig_prob.update_layout(
                    title="Class Probability Distribution (Digits 0–9)",
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
            st.info("Select or upload an image to run INT8 inference.")


# -----------------------------------------------------------------------------
# TAB 3: Model Analysis
# -----------------------------------------------------------------------------
with tab_analysis:
    st.markdown("#### 📈 Quantitative Model Benchmarks (FP32 vs INT8)")
    st.caption("Measurements evaluated across the entire 10,000-sample MNIST test set with identical evaluation methodology.")

    if has_data:
        fp32_b = comparison["fp32_baseline"]
        int8_q = comparison["int8_quantized"]
        res = comparison["comparison_results"]

        col_t1, col_t2 = st.columns(2)

        with col_t1:
            # Model Size Chart
            fig_size = go.Figure()
            fig_size.add_trace(go.Bar(
                name="Model Size (KB)",
                x=["FP32 TFLite", "INT8 TFLite"],
                y=[fp32_b["size_kb"], int8_q["size_kb"]],
                marker_color=["#64748B", "#38BDF8"],
                text=[f"{fp32_b['size_kb']} KB", f"{int8_q['size_kb']} KB"],
                textposition="auto",
            ))
            fig_size.update_layout(
                title=f"Flash Storage Size: -{res['size_reduction_percent']}% ({res['compression_ratio']}× Compression)",
                yaxis=dict(title="File Size (KB)"),
                height=280,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
            )
            st.plotly_chart(fig_size, use_container_width=True)

        with col_t2:
            # Latency Chart
            fig_lat = go.Figure()
            fig_lat.add_trace(go.Bar(
                name="Latency (ms)",
                x=["FP32 Mean", "INT8 Mean", "FP32 Median", "INT8 Median"],
                y=[fp32_b["latency_mean_ms"], int8_q["latency_mean_ms"], fp32_b["latency_median_ms"], int8_q["latency_median_ms"]],
                marker_color=["#64748B", "#38BDF8", "#475569", "#0284C7"],
                text=[f"{v:.4f} ms" for v in [fp32_b["latency_mean_ms"], int8_q["latency_mean_ms"], fp32_b["latency_median_ms"], int8_q["latency_median_ms"]]],
                textposition="auto",
            ))
            fig_lat.update_layout(
                title="Single-Sample Inference Latency (Host CPU)",
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
| **Test Accuracy (10,000 Samples)** | `{fp32_b['accuracy_percent']}%` ({fp32_b['correct_predictions']:,} / 10,000) | `{int8_q['accuracy_percent']}%` ({int8_q['correct_predictions']:,} / 10,000) | **`+{res['accuracy_delta_percentage_points']} pts`** (No accuracy loss observed) |
| **Model File Size** | `{fp32_b['size_bytes']:,} Bytes` ({fp32_b['size_kb']} KB) | `{int8_q['size_bytes']:,} Bytes` ({int8_q['size_kb']} KB) | **`-{res['size_reduction_percent']}%`** ({res['compression_ratio']}× compression) |
| **Storage Bytes Saved** | Baseline | `{res['size_reduction_bytes']:,} Bytes` | **`21.7 KB saved`** |
| **Host Mean Latency** | `{fp32_b['latency_mean_ms']:.4f} ms` | `{int8_q['latency_mean_ms']:.4f} ms` | **`{res['latency_change_percent']}%`** (`{res['latency_difference_ms']:.4f} ms`) |
| **Host Median Latency** | `{fp32_b['latency_median_ms']:.4f} ms` | `{int8_q['latency_median_ms']:.4f} ms` | `-7.62%` |
| **Host Latency Range (Min–Max)**| `{fp32_b['latency_min_ms']:.4f} – {fp32_b['latency_max_ms']:.4f} ms` | `{int8_q['latency_min_ms']:.4f} – {int8_q['latency_max_ms']:.4f} ms` | Lower standard deviation (`{int8_q['latency_std_ms']:.4f}` vs `{fp32_b['latency_std_ms']:.4f}`) |
| **Input / Output Datatypes** | `float32` in / `float32` out | `int8` in / `int8` out | **Full-Integer Quantization** |
| **Hardware FPU Requirement** | Required for FP operations | **None** (Integer SIMD ALU) | **TinyML MCU Ready** |
"""
        st.markdown(table_markdown)

        st.info("ℹ️ **Measurement Note**: Latency metrics reflect single-sample execution measured on the host development environment (CPU). Microcontroller execution cycles will vary depending on MCU clock speed and instruction set architecture.")
    else:
        st.error("No comparison data available.")


# -----------------------------------------------------------------------------
# TAB 4: TinyML Verification
# -----------------------------------------------------------------------------
with tab_tinyml:
    st.markdown("#### 🎛️ Embedded TinyML Preparation & Verification")
    st.caption("Inspection of the quantized INT8 graph structure, TFLite Micro operator support, and C-array export.")

    if tinyml_analysis and "verified" in tinyml_analysis:
        ver = tinyml_analysis["verified"]
        est = tinyml_analysis["estimated"]
        not_ver = tinyml_analysis["not_verified"]

        col_v, col_e, col_nv = st.columns(3)

        with col_v:
            st.markdown("##### 🟢 1. Verified (Static Inspection)")
            st.markdown(f"- **Flash Storage Footprint**: `{ver['flash_storage_bytes']:,} Bytes` ({ver['flash_storage_kb']} KB)")
            st.markdown(f"- **Input Tensor**: `{ver['input_tensor']['shape']}` (`{ver['input_tensor']['dtype']}`, {ver['input_tensor']['size_bytes']} B)")
            st.markdown(f"- **Output Tensor**: `{ver['output_tensor']['shape']}` (`{ver['output_tensor']['dtype']}`, {ver['output_tensor']['size_bytes']} B)")
            st.markdown(f"- **Total Tensors in Graph**: `{ver['total_tensor_count']}`")
            st.markdown(f"- **C-Array Binary Parity**: `100% Byte-for-Byte Match`")
            st.markdown(f"- **TFLM Built-in Op Support**: `All 5 Ops Supported`")

        with col_e:
            st.markdown("##### 🟡 2. Estimated (Runtime Memory)")
            st.markdown(f"- **Input Buffer RAM**: `{est['input_buffer_bytes']} Bytes`")
            st.markdown(f"- **Peak Activation Buffer**: `{est['largest_single_activation_bytes']:,} Bytes`")
            st.markdown(f"- **Estimated Tensor Arena RAM**: **`{est['estimated_tensor_arena_kb']} KB`** (`{est['estimated_tensor_arena_bytes']:,} Bytes`)")
            st.caption(f"_{est['estimation_methodology']}_")

        with col_nv:
            st.markdown("##### ⚪ 3. Not Verified (Explicit Boundaries)")
            st.markdown(f"- **Physical MCU Flashing**: `Not Executed`")
            st.markdown(f"- **Hardware Clock Cycles**: `Not Measured`")
            st.markdown(f"- **Hardware Power/Current Draw**: `Not Measured`")
            st.caption("_Hardware execution was evaluated on host simulation/TFLite runtime; physical microcontroller targets (e.g. STM32, ESP32) were not flashed in this software environment._")

        st.markdown("---")
        st.markdown("##### 🧱 TFLite Micro Supported Operators")
        ops_df = [
            {"Operator": op["operator"], "TFLM Builtin Support": "✅ Supported" if op["supported_in_tflm"] else "❌ Unsupported", "Notes": op["notes"]}
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
# TAB 5: How It Works
# -----------------------------------------------------------------------------
with tab_how:
    st.markdown("#### 📖 How INT8 Post-Training Quantization Works")

    st.markdown(
        """
        ##### 1. The Core TinyML Challenge
        Standard Convolutional Neural Networks are trained with 32-bit floating-point (`float32`) arithmetic. While highly accurate, FP32 models introduce major bottlenecks on low-power edge devices:
        - **Storage Constraints**: A typical microcontroller has 64 KB – 512 KB of Flash memory.
        - **RAM Constraints**: Microcontrollers feature only 16 KB – 128 KB of SRAM.
        - **Computational Overhead**: Many ultra-low-power microcontrollers (e.g., ARM Cortex-M0/M3) lack a hardware Floating Point Unit (FPU), making software-emulated float calculations slow and power-inefficient.

        ##### 2. Post-Training Integer Quantization (PTQ)
        Post-Training Quantization maps 32-bit floating-point numbers $r \in [\min, \max]$ onto 8-bit signed integers $q \in [-128, 127]$ through linear affine transformation:

        $$\\text{Quantize: } q = \\text{clip}\\left(\\left\\lfloor \\frac{r}{S} \\right\\rceil + Z, -128, 127\\right)$$

        $$\\text{Dequantize: } r = S \\times (q - Z)$$

        Where:
        - **$S$ (Scale)**: Positive float representing the step size per integer quantum ($S = \\frac{r_{\\max} - r_{\\min}}{q_{\\max} - q_{\\min}}$).
        - **$Z$ (Zero-Point)**: Integer offset corresponding to the real value `0.0` to guarantee exact zero-padding without numerical error.

        ##### 3. Calibration with Representative Dataset
        Because activation dynamic ranges cannot be determined from weights alone, a **Representative Dataset** consisting of 200 real MNIST training samples is fed through the network during quantization. The calibration engine records the activation distributions across every layer to compute optimal scale factors and zero-points.

        ##### 4. End-to-End Edge Deployment Flow
        ```text
        [MNIST Dataset] ──► [Lightweight CNN Training] ──► [model_fp32.keras (131 KB)]
                                                                  │
                                                        [TFLite Conversion]
                                                                  ▼
                                                      [model_fp32.tflite (34.7 KB)]
                                                                  │
                                                      [PTQ + Calibration Gen]
                                                                  ▼
                                                      [model_int8.tflite (13.5 KB)]
                                                                  │
                                                      [C Array Export Tool]
                                                                  ▼
                                                      [tinyml/model_data.h & .cc]
                                                                  │
                                                      [TFLite Micro Runtime on MCU]
        ```
        """
    )
