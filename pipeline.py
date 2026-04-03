""" bubbaloop_app.pipeline
Main Bubbaloop VLM pipeline — integrates kornia-vlm inference with the Bubbaloop ROS2-compatible camera node interface. 
This module is intentionally framework-agnostic: it works with a plain OpenCV VideoCapture OR a Bubbaloop CameraNode publisher."""
from __future__ import annotations
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from kornia_vlm.applications.scene_understanding import SceneOutput, SceneUnderstanding
from kornia_vlm.applications.visual_qa import VisualQA, QAResult
from kornia_vlm.models.base import BaseVLM, VLMConfig


@dataclass
class PipelineConfig:
    """Runtime configuration for the Bubbaloop VLM pipeline."""

    # Inference settings
    model_variant: str = "qwen2.5-vl-7b-awq"
    backend: str = "tensorrt"          # "pytorch" | "tensorrt"
    precision: str = "fp16"            # "fp16" | "int8"
    device: str = "cuda"

    # Pipeline behaviour
    mode: str = "scene_understanding"  # "scene_understanding" | "visual_qa" | "captioning"
    target_fps: float = 2.0            # inference FPS (not camera FPS)
    max_queue_size: int = 4            # drop frames beyond this
    enable_warmup: bool = True
    warmup_runs: int = 5

    # Application settings
    scene_mode: str = "general"        # "general" | "robot_navigation"
    qa_questions: list = None          # list of recurring Q&A questions

    # Jetson power settings
    jetson_power_mode: int = 0         # 0=MAXN, 1=25W, 2=15W, 3=10W

    def __post_init__(self):
        if self.qa_questions is None:
            self.qa_questions = [
                "Is the path ahead clear?",
                "Are there any humans visible?",
                "Describe the nearest obstacle.",
            ]


class BubbaVLMPipeline:
    """ Real-time VLM inference pipeline for Bubbaloop.
    Runs inference in a dedicated background thread at `target_fps`,
    decoupled from the camera capture rate.  Output callbacks allow
    integration with Bubbaloop's action bus or any other subscriber.
    Parameters:
    model : BaseVLM Pre-loaded VLM model.
    config : PipelineConfig

    Example ->
    >>> pipeline = BubbaVLMPipeline.from_config(PipelineConfig())
    >>> pipeline.register_callback(lambda out: print(out.summary))
    >>> pipeline.start()
    >>> pipeline.push_frame(camera_frame)
    >>> pipeline.stop() """

    def __init__(self, model: BaseVLM, config: Optional[PipelineConfig] = None):
        self.model = model
        self.cfg = config or PipelineConfig()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=self.cfg.max_queue_size)
        self._output_callbacks: list[Callable] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stats = {"processed": 0, "dropped": 0, "errors": 0}
        vlm_cfg = VLMConfig(device=self.cfg.device,dtype=self.cfg.precision,max_new_tokens=256,)
        if self.cfg.mode == "scene_understanding":
            self._app = SceneUnderstanding(
                model=self.model,
                config=vlm_cfg,
                mode=self.cfg.scene_mode,
            )
        elif self.cfg.mode == "visual_qa":
            self._app = VisualQA(model=self.model, config=vlm_cfg)
        else:
            self._app = SceneUnderstanding(model=self.model, config=vlm_cfg)

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "BubbaVLMPipeline":
        """Convenience constructor that loads the model automatically."""
        from kornia_vlm.models.registry import build_model
        vlm_cfg = VLMConfig(device=config.device, dtype=config.precision)
        model = build_model(config.model_variant, config=vlm_cfg)
        model.load()
        if config.enable_warmup:
            model.warmup(n=config.warmup_runs)
        return cls(model=model, config=config)

    def push_frame(self, frame: np.ndarray) -> bool:
        """ Push a camera frame to the inference queue.
        Returns True if queued successfully, False if dropped (queue full). """
        try:
            self._frame_queue.put_nowait(frame)
            return True
        except queue.Full:
            self._stats["dropped"] += 1
            return False

    def register_callback(self, fn: Callable) -> None:
        """Register a callback invoked with each inference result."""
        self._output_callbacks.append(fn)

    def _dispatch(self, result) -> None:
        for cb in self._output_callbacks:
            try:
                cb(result)
            except Exception as e:
                print(f"[Pipeline] Callback error: {e}")

    def _infer_loop(self) -> None:
        interval = 1.0 / self.cfg.target_fps
        while self._running:
            t_start = time.perf_counter()
            try:
                frame = self._frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                if self.cfg.mode == "visual_qa":
                    for question in self.cfg.qa_questions:
                        result = self._app.ask(frame, question)
                        self._dispatch(result)
                else:
                    result = self._app.run(frame)
                    self._dispatch(result)
                self._stats["processed"] += 1
            except Exception as e:
                self._stats["errors"] += 1
                print(f"[Pipeline] Inference error: {e}")
              
            elapsed = time.perf_counter() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self) -> None:
        """Start the background inference thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._thread.start()
        print(f"[Pipeline] Started — mode={self.cfg.mode}, target={self.cfg.target_fps} FPS")

    def stop(self, timeout: float = 5.0) -> None:
        """Gracefully stop the pipeline."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        print(f"[Pipeline] Stopped. Stats: {self._stats}")

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def __enter__(self) -> "BubbaVLMPipeline":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
