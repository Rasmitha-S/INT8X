# PS09: INT8 Quantized CNN Deployment for Resource-Constrained TinyML Devices

An end-to-end TinyML pipeline demonstrating full-integer Post-Training Quantization (PTQ) of a lightweight Convolutional Neural Network (CNN) on the MNIST dataset, achieving a **61.10% reduction in model size** while **preserving 100% of baseline accuracy**.

---

## 1. Project Overview

### Problem Statement
Resource-constrained microcontrollers (e.g., ARM Cortex-M, ESP32, STM32) have severe Flash (<256 KB) and SRAM (<64 KB) limits. Conventional 32-bit floating-point (FP32) neural networks consume excessive memory and require hardware Floating-Point Units (FPUs) that are absent or power-prohibitive on ultra-low-power edge nodes.

### Technical Solution
By applying **Full-Integer Post-Training Quantization (PTQ)** using TensorFlow Lite with representative dataset calibration, weights and activations are mapped to 8-bit signed integers (`int8`). This reduces storage requirements and enables integer-only SIMD ALU execution on microcontrollers without retraining the model.

---

## 2. Pipeline Architecture

```text
       [MNIST Dataset (28x28 Grayscale)]
                      │
                      ▼
       [Lightweight Custom CNN Architecture]
       (2x Conv2D, 2x MaxPool, 2x Dense: 7,834 Params)
                      │
                      ▼
       [FP32 Model Training & Baseline Evaluation]
       (models/model_fp32.keras: 131.4 KB | model_fp32.tflite: 34.7 KB)
                      │
                      ▼
       [Full-Integer INT8 Post-Training Quantization]
       (Calibration with 200 real MNIST training samples)
                      │
                      ▼
       [Verified INT8 TFLite Model]
       (models/model_int8.tflite: 13.5 KB | int8 in / int8 out)
                      │
                      ▼
       [TinyML C-Array Header & Source Export]
       (tinyml/model_data.h & tinyml/model_data.cc: 13,824 bytes)
                      │
                      ▼
       [Streamlit Cloud Interactive Dashboard]
       (app.py: Dashboard, Live Inference, Benchmarks, TinyML)
```

---

## 3. Verified Benchmark Results

All metrics are measured from actual training, quantization, and evaluation runs over the entire **10,000-sample MNIST test set**. No values are hardcoded or fabricated.

| Benchmark Metric | FP32 TFLite Baseline | INT8 TFLite Quantized | Measured Impact |
|:---|:---|:---|:---|
| **Test Accuracy (10,000 Samples)** | `98.44%` (9,844 / 10,000) | `98.46%` (9,846 / 10,000) | **`+0.02 percentage points`** (*No accuracy loss observed*) |
| **Model Storage Size** | `35,536 Bytes` (34.70 KB) | `13,824 Bytes` (13.50 KB) | **`-61.10%`** (`2.57×` compression) |
| **Flash Bytes Saved** | Baseline | `21,712 Bytes` | **`21.2 KB saved`** |
| **Host Mean Latency** | `0.0133 ms` (13.3 µs) | `0.0098 ms` (9.8 µs) | **`-26.32%`** on host CPU |
| **Host Median Latency** | `0.0105 ms` | `0.0097 ms` | `-7.62%` |
| **Tensor Input / Output Dtypes** | `float32` / `float32` | `int8` / `int8` | **Full-Integer Quantization** |
| **Hardware FPU Requirement** | Required for FP math | **None** (Integer SIMD ALU) | **TinyML MCU Ready** |

> **Note on Latency**: Latency figures represent single-sample execution measured on the host development CPU via `time.perf_counter()` after 20 unmeasured warm-up iterations. Microcontroller clock cycles vary by target architecture.

---

## 4. Repository Structure

```text
├── app.py                       # Single-file Streamlit web application (5 tabs)
├── requirements.txt             # Minimal, pinned Python dependencies
├── README.md                    # Project documentation with verified benchmarks
├── .gitignore                   # Standard Python/ML gitignore
├── .streamlit/
│   └── config.toml              # Streamlit theme & headless server configuration
├── src/
│   ├── __init__.py
│   ├── data.py                  # MNIST loading, normalization, and calibration generator
│   ├── model.py                 # Lightweight TinyML CNN (7,834 parameters)
│   ├── train.py                 # FP32 model training routine
│   ├── quantize.py              # Full-integer INT8 PTQ conversion module
│   ├── inference.py             # Modular TFLite inference engine (FP32 & INT8)
│   ├── evaluate.py              # 10,000-sample benchmark runner & comparison export
│   └── metrics.py               # Metrics persistence and file calculation utilities
├── models/
│   ├── model_fp32.keras         # Trained FP32 Keras model
│   ├── model_fp32.tflite        # Baseline FP32 TFLite model
│   └── model_int8.tflite        # Verified INT8 TFLite model (13.5 KB)
├── results/
│   ├── metrics.json             # Raw measured metrics
│   └── comparison.json          # Structured quantitative comparison data
├── tinyml/
│   ├── model_data.h             # C header array declaration
│   ├── model_data.cc            # C++ source byte array definition (13,824 bytes)
│   ├── model_analysis.json      # Static tensor breakdown and operator support
│   └── verify_tinyml.py         # Static analyzer and C-array exporter
├── assets/
│   └── sample_digits/           # Genuine 28x28 MNIST test sample images (0-9)
└── tests/
    ├── test_env.py              # Environment and import validation
    ├── test_data.py             # Data pipeline & normalization tests
    ├── test_model.py            # CNN architecture & parameter limits (<10k)
    ├── test_fp32_baseline.py    # FP32 model & baseline metrics tests
    ├── test_quantization.py     # INT8 tensor datatype and quantization checks
    ├── test_inference.py        # Inference engine & dynamic scale tests
    ├── test_benchmark.py        # Mathematical comparison integrity tests
    ├── test_tinyml.py           # C-array byte parity and operator support tests
    ├── test_app.py              # Streamlit application & path safety tests
    └── test_smoke.py            # End-to-end digit recognition smoke tests
```

---

## 5. Local Setup & Execution

### Prerequisites
- Python 3.10 or 3.11

### 1. Clone & Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/your-username/PS09-TinyQuant.git
cd PS09-TinyQuant

# Create and activate virtual environment
# Windows:
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 6. Running the Complete Test Suite

Execute all 43 automated unit and integration tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 7. Streamlit Community Cloud Deployment

The application is structured for direct deployment on **Streamlit Community Cloud**:

1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io).
3. Select your repository, specify the `main` branch, and set the Main file path to:
   ```text
   app.py
   ```
4. Click **Deploy**. The application will load all pre-generated models and verified metrics instantly without retraining.

---

## 8. Explicit Limitations & Verification Boundaries

To adhere to scientific rigor, the project maintains strict boundaries:

- **Verified**:
  - Full-integer INT8 quantization of weights and activations.
  - Byte-for-byte parity between `models/model_int8.tflite` and `tinyml/model_data.cc` (13,824 bytes).
  - 100% compatibility of model operators (`CONV_2D`, `MAX_POOL_2D`, `RESHAPE`, `FULLY_CONNECTED`, `SOFTMAX`) with TFLite Micro.
  - Measured 10,000-sample test accuracy and host CPU latency.
- **Estimated**:
  - TFLite Micro Tensor Arena memory footprint: **~14.0 KB** (calculated based on maximum concurrent intermediate activations + runtime metadata).
- **Not Verified**:
  - Physical flashing onto a target microcontroller board (e.g. STM32, ESP32) was not performed.
  - Physical MCU clock cycles and hardware power/current draw were not measured.
