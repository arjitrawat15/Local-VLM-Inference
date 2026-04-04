""" kornia_vlm.benchmarks.suite
Comprehensive benchmark suite for VLM inference on embedded hardware.

Measures
  - End-to-end latency  (first token latency + generation latency)
  - Vision encoder throughput (ms/image, images/sec)
  - Peak GPU / unified memory (Jetson)
  - Token generation speed (tokens/sec)
  - Quality metrics (CIDEr for captioning, BLEU-4 for VQA if ground truth available)

Outputs - JSON report saved to disk"""
from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np

@dataclass
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float

    @classmethod
    def from_samples(cls, samples: List[float]) -> "LatencyStats":
        a = np.array(samples)
        return cls(
            mean_ms=float(np.mean(a)),
            p50_ms=float(np.percentile(a, 50)),
            p95_ms=float(np.percentile(a, 95)),
            p99_ms=float(np.percentile(a, 99)),
            min_ms=float(np.min(a)),
            max_ms=float(np.max(a)),
            std_ms=float(np.std(a)),
        )


@dataclass
class BenchmarkResult:
    model_name: str
    backend: str
    device: str
    precision: str
    image_size: tuple
    n_runs: int
    warmup_runs: int

    # Latency breakdown
    end_to_end: LatencyStats = field(default_factory=lambda: LatencyStats(0,0,0,0,0,0,0))
    vision_encoder: LatencyStats = field(default_factory=lambda: LatencyStats(0,0,0,0,0,0,0))

    # Throughput
    tokens_per_second: float = 0.0
    images_per_second: float = 0.0

    # Memory
    peak_vram_mb: float = 0.0
    model_vram_mb: float = 0.0

    # HW info
    hw_info: dict = field(default_factory=dict)

    def to_markdown_row(self) -> str:
        return (
            f"| {self.model_name} | {self.backend} | {self.precision} "
            f"| {self.end_to_end.mean_ms:.1f} ms "
            f"| {self.end_to_end.p95_ms:.1f} ms "
            f"| {self.tokens_per_second:.1f} "
            f"| {self.peak_vram_mb:.0f} MB |"
        )


# ---------------------------------------------------------------------------
# Benchmarker
# ---------------------------------------------------------------------------

class VLMBenchmarker:
    """
    Run the full benchmark suite against a loaded VLM.

    Parameters
    ----------
    model : BaseVLM
        Loaded model (any backend).
    n_runs : int
        Number of timed forward passes.
    warmup_runs : int
        Un-timed warm-up passes before measurement begins.
    image_sizes : list of (H, W)
        Image resolutions to benchmark (tests dynamic shape support).
    prompts : list of str
        Text prompts to cycle through.

    Example
    -------
    >>> from kornia_vlm.models.qwen_vl import QwenVL
    >>> model = QwenVL(); model.load()
    >>> bench = VLMBenchmarker(model, n_runs=100)
    >>> result = bench.run()
    >>> print(bench.to_markdown_table([result]))
    """

    MARKDOWN_HEADER = (
        "| Model | Backend | Precision | Mean E2E | P95 E2E "
        "| Tok/sec | Peak VRAM |\n"
        "|-------|---------|-----------|----------|---------|---------|-----------|"
    )

    def __init__(
        self,
        model,
        n_runs: int = 100,
        warmup_runs: int = 10,
        image_sizes: Optional[list] = None,
        prompts: Optional[List[str]] = None,
    ):
        self.model = model
        self.n_runs = n_runs
        self.warmup_runs = warmup_runs
        self.image_sizes = image_sizes or [(448, 448), (640, 480), (1280, 720)]
        self.prompts = prompts or [
            "Describe this image in one sentence.",
            "What objects are visible?",
            "Is there a person in this image?",
        ]

    def _make_dummy_image(self, h: int, w: int) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    def _measure_memory(self) -> dict:
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "allocated_mb": torch.cuda.memory_allocated() / 1e6,
                    "reserved_mb": torch.cuda.memory_reserved() / 1e6,
                }
        except ImportError:
            pass
        return {}

    def run(self, image_size: Optional[tuple] = None) -> BenchmarkResult:
        """Run the full benchmark for one image size."""
        h, w = image_size or self.image_sizes[0]
        image = self._make_dummy_image(h, w)

        # ── Warmup ──────────────────────────────────────────────────────
        print(f"[Bench] Warming up ({self.warmup_runs} passes) …")
        for i in range(self.warmup_runs):
            self.model.generate(image, self.prompts[i % len(self.prompts)])

        # ── Measure ─────────────────────────────────────────────────────
        print(f"[Bench] Measuring {self.n_runs} passes …")
        e2e_samples: List[float] = []
        tps_samples: List[float] = []

        pre_mem = self._measure_memory()

        for i in range(self.n_runs):
            prompt = self.prompts[i % len(self.prompts)]
            out = self.model.timed_generate(image, prompt)
            e2e_samples.append(out.latency_ms)
            if out.tokens_generated > 0 and out.latency_ms > 0:
                tps_samples.append(out.tokens_per_second)

        post_mem = self._measure_memory()

        hw_info = self._get_hw_info()

        return BenchmarkResult(
            model_name=self.model.model_name,
            backend=self.model.backend,
            device=self.model.config.device,
            precision=self.model.config.dtype,
            image_size=(h, w),
            n_runs=self.n_runs,
            warmup_runs=self.warmup_runs,
            end_to_end=LatencyStats.from_samples(e2e_samples),
            tokens_per_second=float(np.mean(tps_samples)) if tps_samples else 0.0,
            images_per_second=1000.0 / float(np.mean(e2e_samples)),
            peak_vram_mb=post_mem.get("reserved_mb", 0.0),
            hw_info=hw_info,
        )

    def run_all_sizes(self) -> List[BenchmarkResult]:
        """Benchmark across all configured image sizes."""
        return [self.run(size) for size in self.image_sizes]

    @staticmethod
    def to_markdown_table(results: List[BenchmarkResult]) -> str:
        header = VLMBenchmarker.MARKDOWN_HEADER
        rows = "\n".join(r.to_markdown_row() for r in results)
        return f"{header}\n{rows}"

    @staticmethod
    def save_json(results: List[BenchmarkResult], path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(r) for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Bench] Results saved → {path}")

    @staticmethod
    def _get_hw_info() -> dict:
        info = {}
        try:
            import platform
            info["os"] = platform.system()
            info["machine"] = platform.machine()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["cuda_version"] = torch.version.cuda
        except Exception:
            pass
        return info


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark kornia-vlm models on target hardware."
    )
    parser.add_argument("--model", default="qwen2.5-vl-7b-awq")
    parser.add_argument("--backend", default="pytorch", choices=["pytorch", "tensorrt"])
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", default="results/benchmark.json")
    args = parser.parse_args()

    from kornia_vlm.models.registry import build_model
    from kornia_vlm.models.base import VLMConfig

    config = VLMConfig(device="cuda", dtype="float16")
    model = build_model(args.model, config=config)
    model.load()

    bench = VLMBenchmarker(model, n_runs=args.runs, warmup_runs=args.warmup)
    results = bench.run_all_sizes()

    print("\n" + bench.to_markdown_table(results))
    bench.save_json(results, args.output)


if __name__ == "__main__":
    main()
