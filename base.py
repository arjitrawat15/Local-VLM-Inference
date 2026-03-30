"""
kornia_vlm.models.base
======================
Abstract base class for all Vision-Language Models integrated into kornia-vlm.
Designed to be backend-agnostic (PyTorch eager, ONNX, TensorRT).
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class VLMOutput:
    """Structured output from a VLM inference call."""
    text: str
    latency_ms: float
    tokens_generated: int
    model_name: str
    backend: str
    metadata: dict = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        if self.latency_ms == 0:
            return 0.0
        return (self.tokens_generated / self.latency_ms) * 1000.0

    def __repr__(self) -> str:
        return (
            f"VLMOutput(text='{self.text[:60]}...', "
            f"latency={self.latency_ms:.1f}ms, "
            f"tps={self.tokens_per_second:.1f})"
        )


@dataclass
class VLMConfig:
    """Runtime configuration for VLM inference."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = False
    image_size: tuple[int, int] = (448, 448)
    device: str = "cuda"
    dtype: str = "float16"   # float16 | int8 | int4
    use_flash_attention: bool = True
    num_beams: int = 1


class BaseVLM(abc.ABC):
    """
    Abstract base class for Vision-Language Models in kornia-vlm.

    All concrete VLM implementations must inherit from this class and implement
    the abstract methods. This ensures a consistent interface across different
    models (Qwen-VL, LLaVA, InternVL, etc.) and backends (PyTorch, ONNX,
    TensorRT).

    Example
    -------
    >>> class MyVLM(BaseVLM):
    ...     def load(self): ...
    ...     def preprocess_image(self, image): ...
    ...     def generate(self, image, prompt, config): ...
    ...     def export_onnx(self, path): ...
    """

    def __init__(self, model_name: str, config: Optional[VLMConfig] = None):
        self.model_name = model_name
        self.config = config or VLMConfig()
        self._loaded = False
        self._backend = "pytorch"

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights and tokenizer into memory."""
        ...

    @abc.abstractmethod
    def preprocess_image(self, image: np.ndarray) -> object:
        """
        Preprocess a raw image (H x W x C, uint8 BGR/RGB) into model inputs.

        Parameters
        ----------
        image : np.ndarray
            Raw image array from kornia or OpenCV.

        Returns
        -------
        Preprocessed tensor / dict accepted by the model backbone.
        """
        ...

    @abc.abstractmethod
    def generate(
        self,
        image: np.ndarray,
        prompt: str,
        config: Optional[VLMConfig] = None,
    ) -> VLMOutput:
        """
        Run vision-language inference given an image and a text prompt.

        Parameters
        ----------
        image : np.ndarray
            Raw image (H x W x 3, uint8).
        prompt : str
            Natural language query / instruction.
        config : VLMConfig, optional
            Override runtime config for this call.

        Returns
        -------
        VLMOutput
        """
        ...

    @abc.abstractmethod
    def export_onnx(self, save_path: str, opset: int = 17) -> str:
        """Export the model vision encoder to ONNX for deployment."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def warmup(self, n: int = 3) -> None:
        """
        Run *n* dummy forward passes to warm up CUDA kernels / TRT engines.
        Recommended before benchmarking.
        """
        if not self._loaded:
            raise RuntimeError("Call load() before warmup().")
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_prompt = "Describe the image."
        for _ in range(n):
            self.generate(dummy_image, dummy_prompt)

    def timed_generate(
        self,
        image: np.ndarray,
        prompt: str,
        config: Optional[VLMConfig] = None,
    ) -> VLMOutput:
        """Identical to generate() but always records wall-clock latency."""
        t0 = time.perf_counter()
        out = self.generate(image, prompt, config)
        elapsed = (time.perf_counter() - t0) * 1000
        out.latency_ms = elapsed
        return out

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def backend(self) -> str:
        return self._backend

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model_name}', backend='{self._backend}')"
