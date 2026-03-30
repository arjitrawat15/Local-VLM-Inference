<div align="center">

<img src="https://raw.githubusercontent.com/kornia/kornia/main/docs/source/_static/img/kornia_logo.svg" width="180"/>

# Bubbaloop VLM

### Local Vision-Language Model Inference on Embedded Hardware

**GSoC 2026 Proposal Prototype** — Project: *Local VLM Application on Bubbaloop*
Mentors: Edgar Riba, Miquel Farré &nbsp;|&nbsp; Duration: 350 hours &nbsp;|&nbsp; Difficulty: Medium

<br/>

[![Kornia](https://img.shields.io/badge/kornia-contributor-orange?logo=github)](https://github.com/kornia/kornia)
[![kornia-rs](https://img.shields.io/badge/kornia--rs-contributor-blue?logo=github)](https://github.com/kornia/kornia-rs)
[![PRs Merged](https://img.shields.io/badge/merged%20PRs-3-brightgreen)](https://github.com/kornia)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Rust](https://img.shields.io/badge/Rust-kornia--rs-orange?logo=rust)](https://github.com/kornia/kornia-rs)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Jetson](https://img.shields.io/badge/Hardware-Jetson%20Orin-76b900?logo=nvidia)](https://developer.nvidia.com/embedded/jetson-agx-orin)

<br/>

> *Run a 7B vision-language model locally on a Jetson Orin at 2+ FPS — fully integrated into the Bubbaloop robotics framework and contributed back to kornia-rs.*

</div>

---

## Table of Contents

- [Project Vision](#-project-vision)
- [Why This Matters for Kornia](#-why-this-matters-for-kornia)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [My Open-Source Contributions](#-my-open-source-contributions)
- [Technical Approach](#-technical-approach)
- [Model Selection & Justification](#-model-selection--justification)
- [Inference Optimisation Pipeline](#-inference-optimisation-pipeline)
- [Bubbaloop Integration](#-bubbaloop-integration)
- [kornia-rs Contribution Plan](#-kornia-rs-contribution-plan)
- [Benchmark Methodology](#-benchmark-methodology)
- [12-Week Timeline](#-12-week-timeline)
- [Quickstart](#-quickstart)
- [Running Tests](#-running-tests)
- [Future Work](#-future-work)

---

## 🎯 Project Vision

Modern robotics needs vision intelligence that works *without the cloud* — in a factory, a warehouse, or outdoors where latency and privacy matter.  This project delivers:

1. **A working Bubbaloop application** demonstrating real-time scene understanding, visual Q&A, and image captioning on a Jetson Orin, using a locally-running 7B VLM.
2. **A `kornia-vlm` Python module** with a clean, extensible API so the community can plug in new models and inference backends with a single decorator.
3. **A `kornia-vlm` Rust crate** contributed to `kornia-rs` — exposing VLM inference through the same idiomatic Rust interface the library already uses for geometry and image processing.
4. **Reproducible benchmarks** (latency, throughput, memory) published in the README and CI — directly addressing the community gap identified in [kornia-rs issue #718](https://github.com/kornia/kornia-rs/issues/718), which I am assigned to.

---

## 💡 Why This Matters for Kornia

| Today | After GSoC |
|---|---|
| kornia = geometric transforms + classical CV | kornia = CV + on-device VLM inference |
| No VLM support in kornia-rs | `kornia-vlm` crate: VLM inference in Rust |
| Benchmarks exist but aren't published (#718) | Benchmark dashboard in README + CI |
| Bubbaloop has no VLM demo | Complete scene-understanding robot app |

---

## 🏗 System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        BUBBALOOP VLM PIPELINE                        │
│                                                                      │
│  ┌─────────────┐   ┌──────────────────────────────────────────────┐ │
│  │  Camera /   │──▶│             kornia preprocessing             │ │
│  │  Video feed │   │  • Resize (ONNX-safe, PR #3463)              │ │
│  └─────────────┘   │  • Normalise / colour-space convert          │ │
│                    │  • Augmentation (optional)                    │ │
│                    └────────────────────┬─────────────────────────┘ │
│                                         │                            │
│              ┌──────────────────────────▼──────────────────────┐    │
│              │              VLM BACKBONE (Python)              │    │
│              │                                                  │    │
│              │   Vision Encoder ──▶  Cross-Attention ──▶  LLM  │    │
│              │   (ViT / CLIP)         (Qwen2.5 rope)    (7B)   │    │
│              │                                                  │    │
│              │   Backend options:                               │    │
│              │   [PyTorch FP16] [ONNX FP16] [TensorRT FP16/INT8]│    │
│              └────────────────────────┬─────────────────────────┘   │
│                                        │                             │
│      ┌─────────────────────────────────▼──────────────────────────┐ │
│      │                    APPLICATION LAYER                        │ │
│      │                                                             │ │
│      │  SceneUnderstanding │  VisualQA  │  ImageCaptioning         │ │
│      │  (structured JSON)  │  (session) │  (CIDEr-optimised)       │ │
│      └───────────────┬─────────────────────────────────────────────┘ │
│                      │                                               │
│      ┌───────────────▼───────────────────────────┐                  │
│      │           BUBBALOOP ACTION BUS             │                  │
│      │  Nav planner │ Manip controller │ UI/HMI   │                  │
│      └────────────────────────────────────────────┘                  │
└──────────────────────────────────────────────────────────────────────┘

                    kornia-rs (Rust) layer
┌──────────────────────────────────────────────────────────────────────┐
│  kornia-vlm crate:  VlmBackend trait  +  QwenVlBackend               │
│  OnnxVlBackend (ONNX Runtime Rust bindings)                          │
│  Benchmarks: bench_vlm_encode, bench_vlm_generate (Criterion.rs)    │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Flow (Single Frame)

```
camera frame (BGR)
       │
       ▼  kornia.geometry.transform.resize()   ← ONNX-safe (PR #3463)
 [448×448 RGB float16]
       │
       ▼  QwenVL.preprocess_image()
 [processor tokens + pixel_values tensor]
       │
       ▼  TensorRT FP16 vision encoder   (< 15 ms on Orin)
 [image_features: 1 × 256 × 1536]
       │
       ▼  Qwen2.5 LLM decode (KV-cache, greedy)
 [text tokens]
       │
       ▼  JSON parser / fallback
 SceneOutput { objects, relationships, summary }
       │
       ▼  Bubbaloop callback
```

---

## 📁 Repository Structure

```
bubbaloop-vlm/
│
├── kornia_vlm/                    # Python package — contributed to kornia
│   ├── models/
│   │   ├── base.py                # Abstract BaseVLM + VLMConfig + VLMOutput
│   │   ├── qwen_vl.py             # Qwen2.5-VL (3B / 7B / 7B-AWQ)
│   │   └── registry.py            # Model registry (register_vlm decorator)
│   │
│   ├── backends/
│   │   ├── pytorch.py             # Eager PyTorch backend
│   │   ├── onnx.py                # ONNX Runtime backend
│   │   └── tensorrt.py            # TensorRT EP (FP16 + INT8 + DLA)
│   │
│   ├── applications/
│   │   ├── scene_understanding.py # Structured scene analysis → JSON
│   │   ├── visual_qa.py           # Single-shot + multi-turn VQA sessions
│   │   └── image_captioning.py    # CIDEr-aware captioning pipeline
│   │
│   └── benchmarks/
│       └── suite.py               # Full benchmark + Markdown table export
│
├── bubbaloop_app/
│   └── pipeline.py                # Real-time threaded inference pipeline
│
├── kornia_rs_vlm/                 # Rust crate skeleton — kornia-rs contribution
│   ├── src/
│   │   ├── lib.rs
│   │   ├── backend.rs             # VlmBackend trait
│   │   └── qwen.rs                # QwenVlBackend (ONNX Runtime Rust)
│   ├── benches/
│   │   └── bench_vlm.rs           # Criterion benchmarks
│   └── Cargo.toml
│
├── scripts/
│   ├── export_onnx.py             # Export vision encoder → ONNX
│   ├── optimize_tensorrt.py       # Build TRT engine with INT8 calibration
│   └── jetson_demo.py             # Full end-to-end Jetson demo
│
├── tests/
│   └── test_models.py             # Unit tests (GPU-free, mock-based)
│
├── configs/
│   └── jetson_orin.yaml           # Jetson Orin production config
│
└── results/                       # Benchmark JSON + Markdown tables
```

---

## 🔧 My Open-Source Contributions

These merged PRs demonstrate my understanding of the kornia codebase and my
ability to contribute production-quality code to both the Python and Rust
repositories.

---

### ✅ [kornia-rs PR #722](https://github.com/kornia/kornia-rs/pull/722) — Fix ICP Test Assertions with Geometric Metrics
**Merged Feb 20, 2025** &nbsp;·&nbsp; Reviewer: `cjpurackal`

**Problem:** The `test_icp_vanilla` test ran the ICP (Iterative Closest Point) algorithm
but its assertion logic had been commented out due to numerical instability when comparing
rotation matrices element-by-element. The algorithm ran, but correctness was never verified.

**My Solution:** Replaced brittle element-wise comparisons with three meaningful geometric
error metrics that directly measure transformation accuracy:

| Metric | Formula | Threshold |
|--------|---------|-----------|
| Angular rotation error | `arccos((trace(R_err) - 1) / 2)` | < 0.01 rad |
| L2 translation error | `‖t_est - t_gt‖₂` | < 0.01 |
| RMSE convergence | Final RMSE of aligned point clouds | < 0.05 |

This approach is geometrically principled — the same philosophy I will apply to VLM
output validation in this GSoC project (e.g., IoU-based bbox assertions rather than
raw coordinate comparisons).

```rust
// Before: commented-out, fragile
// assert_eq!(rotation, expected_rotation);   // ← unstable

// After: geometric error metrics
let angle_error = ((r_error.trace() - 1.0) / 2.0).acos();
assert!(angle_error < 0.01, "Rotation error too large: {}", angle_error);
let t_error = (t_estimated - t_ground_truth).norm();
assert!(t_error < 0.01, "Translation error too large: {}", t_error);
```

---

### ✅ [kornia PR #3463](https://github.com/kornia/kornia/pull/3463) — Fix Resize ONNX Export Compatibility
**Merged Jan 2025** &nbsp;·&nbsp; Reviewers: `edgarriba`, `sidd-27` &nbsp;·&nbsp; `+226 / -28`

**Problem:** `torch.export` and ONNX tracing raised a hard error when encountering
the `perform_keep_shape_image` decorator on kornia's `Resize` operator. The decorator
performed shape bookkeeping logic incompatible with tracing mode, blocking ONNX export
even for simple resize operations.

**My Solution:** Added export-mode detection guards that bypass shape logic during
ONNX tracing while preserving the original behaviour in eager mode (fully backward compatible):

```python
# kornia/utils/image.py  (my fix)
def perform_keep_shape_image(f):
    @wraps(f)
    def wrapper(input, *args, **kwargs):
        # Guard: skip shape logic during ONNX export / JIT tracing
        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
            return f(input, *args, **kwargs)
        # Original shape-bookkeeping logic (eager mode only)
        ...
    return wrapper
```

Added tests covering fixed-size resize, upscaling/downscaling, multiple interpolation
modes (bilinear, nearest), and a PyTorch-vs-ONNXRuntime output comparison with tolerance 1e-4.

**Direct relevance to this GSoC:** This fix is *essential* for exporting the Qwen-VL
vision encoder to ONNX — without it, any kornia preprocessing in the vision pipeline
would have caused silent export failures.

---

### ✅ [kornia PR #3537](https://github.com/kornia/kornia/pull/3537) — Multi-format Bounding Box Support in `mean_iou_bbox`
**Merged Feb 17, 2025** &nbsp;·&nbsp; Reviewers: `sidd-27`, `edgarriba` &nbsp;·&nbsp; `+139 / -14`

**Problem:** `mean_iou_bbox` only accepted `xyxy` format bounding boxes, forcing users
to manually convert when working with detection outputs in `xywh` (COCO format) or
`cxcywh` (YOLO format).

**My Solution:** Added a `box_format` parameter and a `_convert_boxes_to_xyxy()` helper:

```python
# kornia/metrics/mean_iou.py  (my addition)
def mean_iou_bbox(
    boxes1: Tensor,
    boxes2: Tensor,
    box_format: str = "xyxy",   # NEW: "xyxy" | "xywh" | "cxcywh"
) -> Tensor:
```

| Format | Convention | Use case |
|--------|-----------|---------|
| `xyxy` | `(x1, y1, x2, y2)` | Original (backward compatible) |
| `xywh` | `(x, y, w, h)` | COCO detection datasets |
| `cxcywh` | `(cx, cy, w, h)` | YOLO detection models |

**Relevance to GSoC:** The VLM in this project outputs bounding boxes in normalised
`cxcywh` format (Qwen-VL's native grounding output). My `mean_iou_bbox` fix allows
directly computing IoU-based quality metrics on VLM grounding outputs without any
format conversion layer.

---

### 🚧 [kornia-rs Issue #718](https://github.com/kornia/kornia-rs/issues/718) — Publish Benchmark Results in README/Docs
**Assigned to me · In progress**

The `kornia-rs` library has comprehensive benchmarks (`bench_histogram`, `bench_io`,
`bench_resize`, `bench_warp_affine`) but results aren't published anywhere visible.
For a performance-focused Rust library, benchmark comparisons against OpenCV, Pillow,
and torchvision are a key selling point. I am adding:

- A `Benchmarks` section to the `kornia-rs` README with key results
- CI-based benchmark tracking (Criterion + GitHub Pages)
- Comparisons against OpenCV-Python, Pillow, torchvision for common operations

The same infrastructure will be extended to `bench_vlm_encode` and `bench_vlm_generate`
as part of this GSoC project.

---

## 🧠 Technical Approach

### Why Qwen2.5-VL?

After evaluating seven open VLMs for embedded deployment:

| Model | Params | Min VRAM | Orin Latency* | Supports ONNX | Notes |
|-------|--------|----------|---------------|---------------|-------|
| LLaVA-1.5 | 7B | 14 GB | ~1.2 FPS | Partial | CLIP encoder |
| InternVL2 | 4B | 8 GB | ~1.8 FPS | Yes | Strong OCR |
| **Qwen2.5-VL-7B-AWQ** | **7B int4** | **6 GB** | **~2.4 FPS** | **Yes** | **Best overall** |
| MiniCPM-V | 3B | 6 GB | ~3.1 FPS | Limited | Weaker grounding |
| PaliGemma2 | 3B | 6 GB | ~2.8 FPS | Yes | No chat template |
| moondream2 | 1.8B | 3 GB | ~7 FPS | Yes | Limited capability |

\* *Estimated on Jetson AGX Orin 32 GB, FP16, 448×448 input*

**Qwen2.5-VL** wins on: (1) unified memory fit with AWQ int4, (2) native dynamic
resolution ViT avoiding resize overhead, (3) strong spatial grounding (bounding box
output in natural language), (4) active ONNX export support matching our PR #3463 work.

---

## ⚙️ Inference Optimisation Pipeline

```
 Hugging Face weights (BF16, ~14 GB)
          │
          ▼  AWQ int4 quantisation (autoawq)
 Quantised weights (~3.5 GB, fits Orin 8 GB)
          │
          ▼  torch.export / torch.onnx.export (opset 17)
          │  ← ONNX-safe thanks to kornia PR #3463
 Vision encoder ONNX (~180 MB)
          │
          ▼  TensorRT builder (FP16 engine, cached .trt)
 TRT engine (< 100 MB, ~8 ms/image on Orin)
          │
          ┌──────────────────────────────────┐
          │   Runtime: ORT TensorrtEP        │
          │   Static KV-cache (decode)       │
          │   Flash Attention 2 (if avail)   │
          │   Jetson MAXN power mode         │
          └──────────────────────────────────┘
```

**Expected latency budget on Jetson AGX Orin (32 GB):**

| Stage | PyTorch FP16 | ONNX FP16 | TRT FP16 | TRT INT8 |
|-------|-------------|-----------|----------|---------|
| Image preprocess | 2 ms | 1 ms | 1 ms | 1 ms |
| Vision encoder | 45 ms | 22 ms | 12 ms | 8 ms |
| LLM decode (64 tok) | 380 ms | 380 ms | 320 ms | 280 ms |
| **Total E2E** | **~430 ms** | **~405 ms** | **~335 ms** | **~290 ms** |
| **FPS** | **~2.3** | **~2.5** | **~3.0** | **~3.4** |

---

## 🤖 Bubbaloop Integration

The pipeline connects to Bubbaloop's camera node via a push/pull queue:

```python
from bubbaloop_app.pipeline import BubbaVLMPipeline, PipelineConfig

cfg = PipelineConfig(
    model_variant="qwen2.5-vl-7b-awq",
    mode="scene_understanding",
    scene_mode="robot_navigation",
    target_fps=2.0,
)

pipeline = BubbaVLMPipeline.from_config(cfg)
pipeline.register_callback(lambda result: action_bus.publish(result.to_dict()))

with pipeline:
    for frame in camera_node.stream():
        pipeline.push_frame(frame)
```

**Three application modes:**

| Mode | Use case | Output |
|------|---------|--------|
| `scene_understanding` | What is in front of the robot? | JSON: objects, relations, summary |
| `visual_qa` | Operator asks free-form questions | Stateful multi-turn Q&A |
| `captioning` | Log / report generation | Natural language description |

---

## 🦀 kornia-rs Contribution Plan

The Rust crate `kornia-vlm` will be structured to match the existing `kornia-rs` patterns:

```rust
// kornia_rs_vlm/src/backend.rs
pub trait VlmBackend: Send + Sync {
    fn encode_image(&self, image: &Image<f32, 3>) -> Result<Tensor>;
    fn generate(&self, features: &Tensor, prompt: &str, config: &VlmConfig)
        -> Result<String>;
    fn name(&self) -> &str;
}

// kornia_rs_vlm/src/qwen.rs
pub struct QwenVlBackend {
    session: ort::Session,   // ONNX Runtime Rust bindings
    config: VlmConfig,
}

impl VlmBackend for QwenVlBackend {
    fn encode_image(&self, image: &Image<f32, 3>) -> Result<Tensor> {
        // Uses kornia-rs image utilities (same as bench_resize etc.)
        let resized = kornia::geometry::transform::resize(image, [448, 448])?;
        let input = ndarray_from_kornia_image(&resized)?;
        let outputs = self.session.run(ort::inputs!["pixel_values" => input]?)?;
        Ok(outputs["image_features"].try_extract_tensor()?.into_owned())
    }
    // ...
}
```

**Benchmark additions to kornia-rs** (connecting to issue #718):

```rust
// kornia_rs_vlm/benches/bench_vlm.rs
fn bench_vlm_encode(c: &mut Criterion) {
    let backend = QwenVlBackend::new("qwen2.5-vl-7b-awq_vision_encoder.onnx").unwrap();
    let image = Image::<f32, 3>::new(ImageSize { width: 640, height: 480 }, ...).unwrap();

    c.bench_function("vlm_encode_640x480", |b| {
        b.iter(|| backend.encode_image(black_box(&image)).unwrap())
    });
}
```

---

## 📊 Benchmark Methodology

Following the approach from [kornia-rs issue #718](https://github.com/kornia/kornia-rs/issues/718)
(which I am assigned to), all benchmarks will be:

1. **Reproducible** — fixed random seed, 200 timed runs, 20 warm-up runs discarded
2. **Statistically sound** — report mean, P50, P95, P99, std-dev
3. **Hardware-labelled** — Jetson AGX Orin 32 GB, JetPack 6.0, MAXN power mode
4. **Comparative** — PyTorch FP16 vs ONNX FP16 vs TRT FP16 vs TRT INT8

Results are auto-exported to Markdown tables and embedded in this README via CI.

```
| Model              | Backend    | Precision | Mean E2E | P95 E2E | Tok/sec | Peak VRAM |
|--------------------|------------|-----------|----------|---------|---------|-----------|
| qwen2.5-vl-7b-awq  | pytorch    | fp16      | 428 ms   | 512 ms  | 18.2    | 6800 MB   |
| qwen2.5-vl-7b-awq  | onnx       | fp16      | 398 ms   | 470 ms  | 19.6    | 6500 MB   |
| qwen2.5-vl-7b-awq  | tensorrt   | fp16      | 332 ms   | 390 ms  | 23.5    | 6200 MB   |
| qwen2.5-vl-7b-awq  | tensorrt   | int8      | 288 ms   | 340 ms  | 27.1    | 4800 MB   |
```
*(Target numbers — actual results to be filled during GSoC)*

---

## 📅 12-Week Timeline

### Phase 1 — Foundation & ONNX Pipeline (Weeks 1–4)

**Week 1 — Environment & Baseline**
- [ ] Set up Jetson Orin dev environment (JetPack 6.0, CUDA 12.x, TensorRT 10.x)
- [ ] Reproduce baseline Qwen2.5-VL-7B-AWQ inference with HuggingFace Transformers
- [ ] Write `BaseVLM` abstract class and `VLMConfig` / `VLMOutput` data structures
- [ ] Profile memory and latency baseline using `torch.profiler`

**Week 2 — ONNX Vision Encoder Export**
- [ ] Export Qwen2.5-VL vision encoder to ONNX opset 17 (building on PR #3463 work)
- [ ] Write `export_onnx.py` with dynamic shape support and verification step
- [ ] Validate ONNX output vs PyTorch within tolerance 1e-4
- [ ] Add `onnx.py` backend with `OnnxVisionEncoder` class

**Week 3 — TensorRT Optimisation**
- [ ] Build TensorRT FP16 engine with ORT `TensorrtExecutionProvider`
- [ ] Implement engine caching (`/tmp/trt_engines/`) to avoid re-compilation
- [ ] Benchmark PyTorch vs ONNX vs TRT latency across image sizes
- [ ] Add DLA-offload option for Jetson Orin DLA cores

**Week 4 — First Community Contribution**
- [ ] Open PR to kornia adding `kornia_vlm.models.base` and `kornia_vlm.models.qwen_vl`
- [ ] Write unit tests (GPU-free, mock-based) covering all public interfaces
- [ ] Address review feedback from mentors (Edgar Riba, Miquel Farré)
- [ ] Document ONNX export + TRT path in `docs/`

---

### Phase 2 — Applications & Bubbaloop Integration (Weeks 5–8)

**Week 5 — Scene Understanding**
- [ ] Implement `SceneUnderstanding` with structured JSON output schema
- [ ] Add graceful fallback for non-JSON model outputs (edge case handling)
- [ ] Test on COCO validation images; compute recall@k for object categories
- [ ] Implement `robot_navigation` mode with obstacle detection prompt

**Week 6 — Visual Q&A & Multi-turn Sessions**
- [ ] Implement single-shot `VisualQA.ask()` with latency tracking
- [ ] Implement `VQASession` with dialogue history and image feature caching
- [ ] Benchmark session Q&A latency vs. fresh encode (target: 60 % reduction)
- [ ] Add `ImageCaptioning` application with CIDEr self-consistency scoring

**Week 7 — Bubbaloop Integration**
- [ ] Integrate `BubbaVLMPipeline` with Bubbaloop's camera node interface
- [ ] Implement threaded frame queue with configurable `target_fps` throttling
- [ ] Add callback-based output routing to Bubbaloop action bus
- [ ] Record demo video: robot scene understanding at 2 FPS on Jetson Orin

**Week 8 — End-to-end Testing & Hardening**
- [ ] Write integration tests with mock camera feed
- [ ] Test memory stability over 1-hour continuous run (no OOM / leaks)
- [ ] Handle edge cases: blurry frames, dark images, no objects detected
- [ ] First draft of `PipelineConfig` YAML loader with validation

---

### Phase 3 — kornia-rs & Benchmarks (Weeks 9–12)

**Week 9 — kornia-rs VLM Crate Skeleton**
- [ ] Create `kornia-vlm` Rust crate with `VlmBackend` trait
- [ ] Implement `QwenVlBackend` using ONNX Runtime Rust bindings (`ort` crate)
- [ ] Write `bench_vlm_encode` Criterion benchmark
- [ ] Open PR to kornia-rs for crate skeleton + trait definition

**Week 10 — Benchmark Infrastructure (Issue #718)**
- [ ] Run all existing `kornia-rs` benchmarks on Jetson hardware
- [ ] Compare `bench_resize` against OpenCV-Python, Pillow, torchvision
- [ ] Add `bench_vlm_encode` and `bench_vlm_generate` to the suite
- [ ] Publish results to README benchmark table (resolving issue #718)

**Week 11 — CI Integration & Documentation**
- [ ] Add GitHub Actions workflow for benchmark regression tracking
- [ ] Write Criterion + GitHub Pages publishing pipeline
- [ ] Complete API documentation for all public-facing `kornia_vlm` symbols
- [ ] Write architecture diagram and integration guide for Bubbaloop docs

**Week 12 — Final Deliverables**
- [ ] Polish and merge all outstanding PRs (kornia + kornia-rs)
- [ ] Record final demo video (scene understanding + VQA on live camera)
- [ ] Write GSoC final report with benchmark tables and lessons learned
- [ ] Open issues / roadmap for post-GSoC community contributions

---

## 🚀 Quickstart

```bash
# Clone
git clone https://github.com/arjitrawat15/bubbaloop-vlm
cd bubbaloop-vlm

# Install (CPU-only for dev/testing)
pip install -e ".[dev]"

# Install with Jetson / GPU support
pip install -e ".[jetson]"

# Run unit tests (no GPU required)
pytest tests/ -v

# Export vision encoder to ONNX
python scripts/export_onnx.py --model qwen2.5-vl-7b-awq --verify

# Run scene understanding on a single image
python scripts/jetson_demo.py --image sample.jpg --mode scene_understanding

# Run on live camera
python scripts/jetson_demo.py --camera 0 --mode visual_qa --fps 2.0

# Run benchmarks
python -m kornia_vlm.benchmarks.suite --model qwen2.5-vl-7b-awq \
    --backend tensorrt --runs 200 --output results/jetson_orin.json
```

---

## 🧪 Running Tests

```bash
# All tests (no GPU needed — uses mocks)
pytest tests/ -v --cov=kornia_vlm --cov-report=term-missing

# Individual test modules
pytest tests/test_models.py -v        # Model abstractions
pytest tests/test_backends.py -v      # Backend switching
pytest tests/test_applications.py -v  # Scene understanding, VQA
```

---

## 🔭 Future Work

Beyond GSoC, the `kornia-vlm` infrastructure enables:

- **Additional models** — InternVL2, PaliGemma2, moondream2 (register with `@register_vlm`)
- **INT8 calibration pipeline** — automated calibration dataset generation from robot logs
- **Streaming decode** — token-by-token streaming for lower perceived latency
- **Multi-camera fusion** — fuse descriptions from multiple Bubbaloop cameras
- **Active perception** — VLM-guided camera pan/tilt for object search

---

## 📄 License

Apache 2.0 — matching the kornia project license.

---

<div align="center">

**Built for [Google Summer of Code 2026](https://summerofcode.withgoogle.com/) with [Kornia](https://github.com/kornia/kornia)**

*Arjit Rawat &nbsp;·&nbsp; GitHub: [@arjitrawat15](https://github.com/arjitrawat15)*

</div>
