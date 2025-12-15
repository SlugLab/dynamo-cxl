// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the CXL Expert Manager.
//!
//! This module exposes the Rust CXL Expert Manager to Python, providing
//! high-performance expert weight management for MoE models with CXL memory tiering.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use dynamo_llm::kv_router::cxl_expert_manager::{
    CxlExpertManager as RustCxlExpertManager, CxlExpertManagerConfig, ExpertLocation,
};
use dynamo_llm::kv_router::protocols::{ExternalSequenceBlockHash, LocalBlockHash};

/// Python wrapper for CXL Expert Manager configuration
#[pyclass(name = "CxlExpertManagerConfig")]
#[derive(Clone)]
pub struct PyCxlExpertManagerConfig {
    #[pyo3(get, set)]
    pub num_experts: u32,
    #[pyo3(get, set)]
    pub num_moe_layers: u32,
    #[pyo3(get, set)]
    pub expert_weight_size: u64,
    #[pyo3(get, set)]
    pub gpu_expert_capacity: u64,
    #[pyo3(get, set)]
    pub cxl_expert_capacity: u64,
    #[pyo3(get, set)]
    pub window_size: u32,
    #[pyo3(get, set)]
    pub cxl_bandwidth_gbps: f64,
    #[pyo3(get, set)]
    pub max_gpu_experts: usize,
    #[pyo3(get, set)]
    pub enable_p2p_dma: bool,
    #[pyo3(get, set)]
    pub qos_priority_levels: u32,
}

#[pymethods]
impl PyCxlExpertManagerConfig {
    #[new]
    #[pyo3(signature = (
        num_experts = 128,
        num_moe_layers = 32,
        expert_weight_size = 268435456,
        gpu_expert_capacity = 85899345920,
        cxl_expert_capacity = 549755813888,
        window_size = 16,
        cxl_bandwidth_gbps = 128.0,
        max_gpu_experts = 32,
        enable_p2p_dma = true,
        qos_priority_levels = 4
    ))]
    pub fn new(
        num_experts: u32,
        num_moe_layers: u32,
        expert_weight_size: u64,
        gpu_expert_capacity: u64,
        cxl_expert_capacity: u64,
        window_size: u32,
        cxl_bandwidth_gbps: f64,
        max_gpu_experts: usize,
        enable_p2p_dma: bool,
        qos_priority_levels: u32,
    ) -> Self {
        Self {
            num_experts,
            num_moe_layers,
            expert_weight_size,
            gpu_expert_capacity,
            cxl_expert_capacity,
            window_size,
            cxl_bandwidth_gbps,
            max_gpu_experts,
            enable_p2p_dma,
            qos_priority_levels,
        }
    }

    #[staticmethod]
    pub fn default() -> Self {
        Self {
            num_experts: 128,
            num_moe_layers: 32,
            expert_weight_size: 256 * 1024 * 1024,
            gpu_expert_capacity: 80 * 1024 * 1024 * 1024,
            cxl_expert_capacity: 512 * 1024 * 1024 * 1024,
            window_size: 16,
            cxl_bandwidth_gbps: 128.0,
            max_gpu_experts: 32,
            enable_p2p_dma: true,
            qos_priority_levels: 4,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CxlExpertManagerConfig(num_experts={}, num_moe_layers={}, max_gpu_experts={}, window_size={})",
            self.num_experts, self.num_moe_layers, self.max_gpu_experts, self.window_size
        )
    }
}

impl From<&PyCxlExpertManagerConfig> for CxlExpertManagerConfig {
    fn from(config: &PyCxlExpertManagerConfig) -> Self {
        CxlExpertManagerConfig {
            num_experts: config.num_experts,
            num_moe_layers: config.num_moe_layers,
            expert_weight_size: config.expert_weight_size,
            gpu_expert_capacity: config.gpu_expert_capacity,
            cxl_expert_capacity: config.cxl_expert_capacity,
            window_size: config.window_size,
            cxl_bandwidth_gbps: config.cxl_bandwidth_gbps,
            max_gpu_experts: config.max_gpu_experts,
            enable_p2p_dma: config.enable_p2p_dma,
            qos_priority_levels: config.qos_priority_levels,
        }
    }
}

/// Python wrapper for CXL Expert Manager
#[pyclass(name = "CxlExpertManager")]
pub struct PyCxlExpertManager {
    inner: Arc<RustCxlExpertManager>,
    cancel_token: CancellationToken,
}

#[pymethods]
impl PyCxlExpertManager {
    /// Create a new CXL Expert Manager
    #[new]
    #[pyo3(signature = (config = None))]
    pub fn new(config: Option<PyCxlExpertManagerConfig>) -> PyResult<Self> {
        let cancel_token = CancellationToken::new();
        let rust_config = config
            .as_ref()
            .map(|c| c.into())
            .unwrap_or_default();

        let manager = RustCxlExpertManager::new(rust_config, cancel_token.clone());

        Ok(Self {
            inner: Arc::new(manager),
            cancel_token,
        })
    }

    /// Register an expert in the system
    #[pyo3(signature = (expert_id, layer_id, weight_size_bytes, initial_location = "gpu"))]
    pub fn register_expert(
        &self,
        expert_id: u32,
        layer_id: u32,
        weight_size_bytes: u64,
        initial_location: &str,
    ) -> PyResult<()> {
        let location = match initial_location {
            "gpu" => ExpertLocation::GpuHbm,
            "cxl" => ExpertLocation::CxlMemory,
            "evicted" => ExpertLocation::Evicted,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid location: {}. Expected 'gpu', 'cxl', or 'evicted'",
                    initial_location
                )))
            }
        };

        self.inner
            .register_expert(expert_id, layer_id, weight_size_bytes, location);
        Ok(())
    }

    /// Request an expert for inference (async)
    pub fn request_expert<'py>(
        &self,
        py: Python<'py>,
        expert_id: u32,
        layer_id: u32,
        priority: u32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manager = Arc::clone(&self.inner);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            manager
                .request_expert(expert_id, layer_id, priority)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
        })
    }

    /// Record a KV delta for windowed checkpointing
    pub fn record_kv_delta(
        &self,
        token_offset: u32,
        layer_id: u32,
        expert_id: u32,
        topk_experts: Vec<u32>,
        gating_scores: Vec<f32>,
        kv_block_hash: u64,
        tokens_hash: u64,
    ) -> PyResult<()> {
        self.inner.record_kv_delta(
            token_offset,
            layer_id,
            expert_id,
            topk_experts,
            gating_scores,
            ExternalSequenceBlockHash(kv_block_hash),
            LocalBlockHash(tokens_hash),
        );
        Ok(())
    }

    /// Force checkpoint commit
    pub fn force_checkpoint(&self) -> PyResult<()> {
        self.inner.force_checkpoint();
        Ok(())
    }

    /// Perform fast recovery (async)
    pub fn fast_recovery<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let recovery_data = self.inner.get_recovery_data();
        let manager = Arc::clone(&self.inner);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            manager
                .fast_recovery(recovery_data)
                .await
                .map(|d| d.as_millis() as f64)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
        })
    }

    /// Get current metrics as a dictionary
    pub fn get_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let metrics = self.inner.get_metrics();
        let dict = PyDict::new(py);

        dict.set_item(
            "total_prefetches",
            metrics
                .total_prefetches
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;
        dict.set_item(
            "total_evictions",
            metrics
                .total_evictions
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;
        dict.set_item(
            "cache_hits",
            metrics
                .cache_hits
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;
        dict.set_item(
            "cache_misses",
            metrics
                .cache_misses
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;
        dict.set_item(
            "total_bytes_transferred",
            metrics
                .total_bytes_transferred
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;
        dict.set_item(
            "avg_transfer_latency_us",
            metrics
                .avg_transfer_latency_us
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;
        dict.set_item(
            "recovery_count",
            metrics
                .recovery_count
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;
        dict.set_item(
            "avg_recovery_time_ms",
            metrics
                .avg_recovery_time_ms
                .load(std::sync::atomic::Ordering::Relaxed),
        )?;

        // Calculate hit rate
        let hits = metrics
            .cache_hits
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let misses = metrics
            .cache_misses
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let hit_rate = if hits + misses > 0.0 {
            hits / (hits + misses)
        } else {
            0.0
        };
        dict.set_item("cache_hit_rate", hit_rate)?;

        Ok(dict)
    }

    /// Shutdown the expert manager
    pub fn shutdown(&self) -> PyResult<()> {
        self.cancel_token.cancel();
        Ok(())
    }

    fn __repr__(&self) -> String {
        let metrics = self.inner.get_metrics();
        let hits = metrics
            .cache_hits
            .load(std::sync::atomic::Ordering::Relaxed);
        let misses = metrics
            .cache_misses
            .load(std::sync::atomic::Ordering::Relaxed);
        format!(
            "CxlExpertManager(cache_hits={}, cache_misses={}, hit_rate={:.2}%)",
            hits,
            misses,
            if hits + misses > 0 {
                (hits as f64 / (hits + misses) as f64) * 100.0
            } else {
                0.0
            }
        )
    }
}

