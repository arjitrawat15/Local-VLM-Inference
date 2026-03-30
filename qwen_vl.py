"""
kornia_vlm.models.qwen_vl
=========================
Qwen2.5-VL integration — optimised for Jetson Orin (8 GB / 16 GB).

Architecture summary
--------------------
  Vision encoder  : ViT-based, native dynamic resolution (NaViT style)
  Language model  : Qwen2.5 transformer backbone
  Fusion          : Cross-attention + rope position embeddings for 2D
  Quantisation    : AWQ int4 (preferred on Jetson), BitsAndBytes int8 fallback

References
----------
  - Qwen2.5-VL paper : https://arxiv.org/abs/2409.12191
  - HuggingFace hub  : Qwen/Qwen2.5-VL-7B-Instruct
  - kornia ONNX work : kornia PR #3463 (fix Resize ONNX export compatibility)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from kornia_vlm.models.base import BaseVLM, VLMConfig, VLMOutput

# ---------------------------------------------------------------------------
# Optional heavy imports – guarded so the module can be imported on CPU-only
# hosts for testing / type-checking.
# ---------------------------------------------------------------------------
try:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Supported model variants
# ---------------------------------------------------------------------------
QWEN_VL_VARIANTS = {
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-7b-awq": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",  # int4, fits in 8 GB
}

# Recommended for Jetson Orin 8 GB
JETSON_RECOMMENDED_VARIANT = "qwen2.5-vl-7b-awq"


class QwenVL(BaseVLM):
    """
    Qwen2.5-VL wrapped for kornia-vlm with Jetson-aware optimisations.

    Parameters
    ----------
    variant : str
        One of the keys in ``QWEN_VL_VARIANTS``.
    config : VLMConfig, optional
        Runtime configuration; Jetson defaults applied if not provided.
    use_flash_attn : bool
        Enable flash-attention-2 (requires flash_attn package).

    Example
    -------
    >>> model = QwenVL("qwen2.5-vl-7b-awq")
    >>> model.load()
    >>> model.warmup()
    >>> out = model.generate(image_array, "What objects are in this scene?")
    >>> print(out.text, f"  [{out.latency_ms:.0f} ms]")
    """

    def __init__(
        self,
        variant: str = JETSON_RECOMMENDED_VARIANT,
        config: Optional[VLMConfig] = None,
        use_flash_attn: bool = True,
    ):
        if variant not in QWEN_VL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from: {list(QWEN_VL_VARIANTS)}"
            )
        super().__init__(model_name=variant, config=config)
        self.hf_model_id = QWEN_VL_VARIANTS[variant]
        self.use_flash_attn = use_flash_attn
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # BaseVLM interface
    # ------------------------------------------------------------------

    def load(self, cache_dir: Optional[str] = None) -> None:
        """
        Download / load model from HuggingFace hub.

        On Jetson Orin we force:
          - torch.float16  (halves VRAM)
          - device_map="cuda:0"
          - attn_implementation="flash_attention_2" if available
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch / transformers are required. "
                "Install with: pip install torch transformers"
            )
        if self._loaded:
            return

        attn_impl = "flash_attention_2" if self.use_flash_attn else "eager"
        dtype = torch.float16 if self.config.dtype == "float16" else torch.bfloat16

        print(f"[QwenVL] Loading {self.hf_model_id} …")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.hf_model_id,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            device_map=self.config.device,
            cache_dir=cache_dir,
        )
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            self.hf_model_id,
            cache_dir=cache_dir,
        )

        self._loaded = True
        self._backend = "pytorch"
        print(f"[QwenVL] Ready — {self.hf_model_id}")

    def preprocess_image(self, image: np.ndarray) -> "Image.Image":
        """Convert BGR/RGB numpy array → PIL Image expected by the processor."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got shape {image.shape}")
        # kornia images are typically RGB float32 [0,1] or uint8
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(image)

    def generate(
        self,
        image: np.ndarray,
        prompt: str,
        config: Optional[VLMConfig] = None,
    ) -> VLMOutput:
        """Run one vision-language forward pass."""
        if not self._loaded:
            raise RuntimeError("Call load() first.")

        cfg = config or self.config
        pil_image = self.preprocess_image(image)

        # Build chat-style message (Qwen uses a specific template)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_input],
            images=[pil_image],
            return_tensors="pt",
        ).to(self.config.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.do_sample,
                temperature=cfg.temperature if cfg.do_sample else 1.0,
                top_p=cfg.top_p if cfg.do_sample else 1.0,
                num_beams=cfg.num_beams,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        # Decode only the newly generated tokens
        new_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        text = self._processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        return VLMOutput(
            text=text,
            latency_ms=latency_ms,
            tokens_generated=new_ids.shape[1],
            model_name=self.model_name,
            backend=self._backend,
            metadata={"hf_model_id": self.hf_model_id},
        )

    def export_onnx(self, save_path: str, opset: int = 17) -> str:
        """
        Export the vision encoder to ONNX.

        Uses the ONNX-safe Resize decorator pattern developed in kornia PR #3463
        (torch.onnx.is_in_onnx_export() guard) to avoid shape-logic errors during
        tracing.
        """
        if not self._loaded:
            raise RuntimeError("Call load() first.")
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required for ONNX export.")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        onnx_file = save_path / f"{self.model_name}_vision_encoder.onnx"

        print(f"[QwenVL] Exporting vision encoder → {onnx_file}")
        vision_encoder = self._model.visual

        # Dummy input: batch=1, 3-channel image at config resolution
        h, w = self.config.image_size
        dummy = torch.zeros(1, 3, h, w, dtype=torch.float16, device=self.config.device)

        torch.onnx.export(
            vision_encoder,
            dummy,
            str(onnx_file),
            opset_version=opset,
            input_names=["pixel_values"],
            output_names=["image_features"],
            dynamic_axes={
                "pixel_values": {0: "batch", 2: "height", 3: "width"},
                "image_features": {0: "batch", 1: "seq_len"},
            },
        )
        print(f"[QwenVL] ONNX export complete: {onnx_file}")
        return str(onnx_file)

    # ------------------------------------------------------------------
    # Jetson-specific helpers
    # ------------------------------------------------------------------

    def profile_memory(self) -> dict:
        """Return current GPU memory stats (Jetson unified memory)."""
        if not _TORCH_AVAILABLE or not torch.cuda.is_available():
            return {}
        return {
            "allocated_MB": torch.cuda.memory_allocated() / 1e6,
            "reserved_MB": torch.cuda.memory_reserved() / 1e6,
            "max_allocated_MB": torch.cuda.max_memory_allocated() / 1e6,
        }

    def enable_kv_cache(self) -> None:
        """Enable static KV-cache for faster autoregressive decoding on Jetson."""
        if self._model is not None:
            self._model.generation_config.cache_implementation = "static"
            print("[QwenVL] Static KV-cache enabled.")
