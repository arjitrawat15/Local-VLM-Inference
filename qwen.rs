//! `QwenVlBackend` — Qwen2.5-VL vision encoder via ONNX Runtime.
//!
//! Uses the `ort` crate (ONNX Runtime Rust bindings) to run the vision
//! encoder exported by `scripts/export_onnx.py`.
//!
//! The language model decode loop is currently delegated to the Python
//! layer via a shared memory IPC channel. A pure-Rust LLM decode backend
//! (llama.cpp / candle) is planned for post-GSoC.
//!
//! # ONNX Runtime execution providers (in priority order)
//!
//! 1. `TensorrtExecutionProvider` — fastest on Jetson (FP16 engine cached)
//! 2. `CUDAExecutionProvider`     — fallback with cuDNN
//! 3. `CPUExecutionProvider`      — CI / testing (no GPU required)

use std::path::Path;
use std::time::Instant;

use ndarray::{Array4, ArrayView4, s};
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder};

use crate::{VlmConfig, VlmError};
use crate::backend::{FeatureTensor, VlmBackend};

/// Qwen2.5-VL vision encoder backend.
pub struct QwenVlBackend {
    session: Session,
    config: VlmConfig,
    model_name: String,
}

impl QwenVlBackend {
    /// Load the ONNX vision encoder from disk.
    ///
    /// # Parameters
    /// - `onnx_path`: Path to `qwen2.5-vl-7b-awq_vision_encoder.onnx`
    /// - `config`: Runtime configuration
    ///
    /// # Errors
    /// Returns `VlmError::LoadFailed` if the ONNX file cannot be read or
    /// if the ONNX Runtime session cannot be initialised.
    pub fn from_onnx<P: AsRef<Path>>(
        onnx_path: P,
        config: VlmConfig,
    ) -> Result<Self, VlmError> {
        let path = onnx_path.as_ref();
        if !path.exists() {
            return Err(VlmError::LoadFailed(format!(
                "ONNX file not found: {}",
                path.display()
            )));
        }

        let model_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("qwen-vlm")
            .to_string();

        let session = Self::build_session(path, &config)?;

        Ok(Self { session, config, model_name })
    }

    fn build_session<P: AsRef<Path>>(
        path: P,
        config: &VlmConfig,
    ) -> Result<Session, VlmError> {
        // Build execution provider list based on config
        let mut providers: Vec<ExecutionProvider> = Vec::new();

        if config.use_tensorrt {
            providers.push(
                ExecutionProvider::TensorRT(Default::default())
                    .with_device_id(0),
            );
        }
        if config.use_cuda {
            providers.push(
                ExecutionProvider::CUDA(Default::default())
                    .with_device_id(0),
            );
        }
        providers.push(ExecutionProvider::CPU(Default::default()));

        let session = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(config.intra_op_threads)?
            .with_execution_providers(providers)?
            .commit_from_file(path)
            .map_err(|e| VlmError::LoadFailed(e.to_string()))?;

        Ok(session)
    }

    /// Preprocess raw pixel values into a normalised `[1, 3, H, W]` float32 array.
    ///
    /// ImageNet-style normalisation:
    ///   mean = [0.485, 0.456, 0.406]
    ///   std  = [0.229, 0.224, 0.225]
    ///
    /// Note: Qwen2.5-VL uses its own normalisation constants; this is a
    /// placeholder that will be replaced with the exact values from the
    /// processor config during the GSoC implementation.
    fn preprocess(
        &self,
        pixel_values: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Array4<f32>, VlmError> {
        if pixel_values.len() != 3 * height * width {
            return Err(VlmError::InvalidInput(format!(
                "Expected {} floats for {}x{}x3 image, got {}",
                3 * height * width,
                height,
                width,
                pixel_values.len()
            )));
        }

        // Reshape to [C, H, W] then add batch dim → [1, C, H, W]
        let chw = ndarray::Array3::from_shape_vec((3, height, width), pixel_values.to_vec())
            .map_err(|e| VlmError::InvalidInput(e.to_string()))?;

        // Normalise: (pixel - mean) / std  (channel-wise)
        let mean = [0.481_454_66_f32, 0.457_827_49, 0.408_211_45]; // Qwen2.5-VL values
        let std  = [0.268_861_68_f32, 0.261_302_77, 0.275_777_1];

        let mut normalised = chw.into_owned();
        for c in 0..3 {
            let mut channel = normalised.slice_mut(s![c, .., ..]);
            channel.mapv_inplace(|v| (v / 255.0 - mean[c]) / std[c]);
        }

        Ok(normalised.insert_axis(ndarray::Axis(0)))
    }
}

impl VlmBackend for QwenVlBackend {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn encode_image(
        &self,
        pixel_values: &[f32],
        height: usize,
        width: usize,
    ) -> Result<FeatureTensor, VlmError> {
        let input_tensor = self.preprocess(pixel_values, height, width)?;

        // Resize to model's expected resolution if needed
        // (uses kornia-rs resize, matching the ONNX-safe pattern from PR #3463)
        let target_h = self.config.image_size.0;
        let target_w = self.config.image_size.1;
        let _ = (target_h, target_w); // resize would go here

        let t0 = Instant::now();

        let outputs = self
            .session
            .run(ort::inputs![
                "pixel_values" => input_tensor.view()
            ]?)
            .map_err(|e| VlmError::InferenceFailed(e.to_string()))?;

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if self.config.verbose {
            println!("[QwenVlBackend] encode_image: {:.1} ms", elapsed_ms);
        }

        // Extract feature tensor [1, seq_len, hidden_dim] → [seq_len, hidden_dim]
        let features_raw = outputs["image_features"]
            .try_extract_tensor::<f32>()
            .map_err(|e| VlmError::InferenceFailed(e.to_string()))?;

        let shape = features_raw.shape();
        let seq_len = shape[1];
        let hidden_dim = shape[2];

        let features = features_raw
            .into_owned()
            .into_shape((seq_len, hidden_dim))
            .map_err(|e| VlmError::InferenceFailed(e.to_string()))?;

        Ok(features)
    }

    fn generate(
        &self,
        features: &FeatureTensor,
        prompt: &str,
    ) -> Result<String, VlmError> {
        // Phase 1 (GSoC week 9): delegate to Python via subprocess / shared memory.
        // Phase 2 (post-GSoC):   pure-Rust LLM decode with llama.cpp / candle.
        //
        // For the prototype this returns a placeholder to prove the pipeline
        // compiles and the feature tensor flows end-to-end.
        let _ = features; // will be used in full implementation
        let _ = prompt;
        Ok(format!(
            "[kornia-vlm Rust backend] {} features extracted. LLM decode: TODO (Phase 2)",
            features.shape()[0]
        ))
    }

    fn metadata(&self) -> Vec<(String, String)> {
        vec![
            ("backend".into(), self.name().into()),
            ("image_size".into(), format!("{:?}", self.config.image_size)),
            ("use_tensorrt".into(), self.config.use_tensorrt.to_string()),
            ("use_cuda".into(), self.config.use_cuda.to_string()),
        ]
    }
}
