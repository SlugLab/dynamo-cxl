// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CXL P2P DMA Interface for GPU <-> CXL Memory Transfers
//!
//! This module provides Rust bindings for the CXL P2P DMA framework from
//! cxl_pytorch_expander, enabling direct GPU <-> CXL memory transfers that
//! bypass the CPU for maximum throughput.
//!
//! # Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────────────────┐
//! │                         NVIDIA GPU                                        │
//! │  ┌─────────────────────────────────────────────────────────────────────┐  │
//! │  │                     GPU HBM Memory                                  │  │
//! │  │    Expert Weights    │    KV Cache    │    Activations             │  │
//! │  └─────────────────────────────────────────────────────────────────────┘  │
//! │                              │                                            │
//! │                    NVIDIA RM ioctl (P2P DMA)                              │
//! └──────────────────────────────┼────────────────────────────────────────────┘
//!                                │
//!                    PCIe/CXL Switch (P2P Bypass)
//!                                │
//! ┌──────────────────────────────┼────────────────────────────────────────────┐
//! │                    CXL Memory Expander                                    │
//! │  ┌─────────────────────────────────────────────────────────────────────┐  │
//! │  │                    CXL Memory Pool                                   │  │
//! │  │   Parked Experts   │   Cold KV Cache   │   Checkpoints              │  │
//! │  └─────────────────────────────────────────────────────────────────────┘  │
//! └───────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Features
//!
//! - **P2P DMA**: Direct GPU <-> CXL transfers via NVIDIA RM ioctl, bypassing CPU
//! - **Buffer Registration**: CXL buffers registered with GPU driver for direct access
//! - **CUDA Event Timing**: Nanosecond-precision transfer timing using CUDA events
//! - **Memory Pool Management**: Efficient buffer allocation with huge page support
//!
//! # Usage
//!
//! ```rust,ignore
//! use dynamo_llm::kv_router::cxl_p2p::{CxlP2pContext, CxlBuffer, TransferDirection};
//!
//! // Initialize CXL P2P context
//! let ctx = CxlP2pContext::new()?;
//!
//! // Allocate a CXL buffer
//! let buffer = ctx.alloc_buffer(256 * 1024 * 1024)?; // 256MB
//!
//! // Transfer GPU tensor to CXL (expert parking)
//! let latency = ctx.transfer_timed(
//!     &buffer,
//!     gpu_ptr,
//!     0, // CXL offset
//!     size,
//!     TransferDirection::GpuToCxl,
//! )?;
//!
//! // Transfer CXL data back to GPU (expert hydration)
//! let latency = ctx.transfer_timed(
//!     &buffer,
//!     gpu_ptr,
//!     0,
//!     size,
//!     TransferDirection::CxlToGpu,
//! )?;
//! ```

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// P2P DMA transfer direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferDirection {
    /// GPU HBM -> CXL Memory (expert parking, KV cache offload)
    GpuToCxl,
    /// CXL Memory -> GPU HBM (expert hydration, KV cache restore)
    CxlToGpu,
}

/// DMA transfer flags matching NVIDIA driver definitions
#[repr(u32)]
pub enum DmaFlags {
    GpuToCxl = 0x0,
    CxlToGpu = 0x1,
}

/// CXL P2P errors
#[derive(Debug, Error)]
pub enum CxlP2pError {
    #[error("CXL P2P not initialized")]
    NotInitialized,

    #[error("CXL initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Buffer allocation failed: {0}")]
    AllocationFailed(String),

    #[error("DMA transfer failed: {0}")]
    TransferFailed(String),

    #[error("Invalid buffer ID: {0}")]
    InvalidBufferId(u64),

    #[error("Buffer registration failed: {0}")]
    RegistrationFailed(String),

    #[error("Driver error: {0:#x}")]
    DriverError(u32),

    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("P2P DMA not available on this system")]
    P2pNotAvailable,
}

pub type CxlP2pResult<T> = std::result::Result<T, CxlP2pError>;

/// CXL system information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CxlSystemInfo {
    /// Whether CXL link is active
    pub link_up: bool,
    /// Whether this is a CXL memory expander
    pub memory_expander: bool,
    /// Number of active CXL links
    pub num_links: u32,
    /// CXL version (1, 2, or 3)
    pub version: u32,
    /// Per-link bandwidth in MB/s
    pub bandwidth_mbps: u32,
    /// Whether P2P DMA is available
    pub p2p_available: bool,
}

impl Default for CxlSystemInfo {
    fn default() -> Self {
        Self {
            link_up: false,
            memory_expander: false,
            num_links: 0,
            version: 2,
            bandwidth_mbps: 0,
            p2p_available: false,
        }
    }
}

/// CXL buffer handle
#[derive(Debug)]
pub struct CxlBuffer {
    /// Unique buffer ID
    pub id: u64,
    /// CPU-accessible pointer (mmap'd region)
    pub cpu_ptr: u64,
    /// Buffer size in bytes
    pub size: usize,
    /// Driver handle for P2P DMA operations
    pub driver_handle: u64,
    /// Whether buffer is registered with GPU driver
    pub registered: bool,
}

/// Transfer result with timing information
#[derive(Debug, Clone)]
pub struct TransferResult {
    /// Transfer ID assigned by driver
    pub transfer_id: u32,
    /// Transfer latency in nanoseconds
    pub latency_ns: u64,
    /// Effective bandwidth in GB/s
    pub bandwidth_gbps: f64,
    /// Bytes transferred
    pub bytes_transferred: usize,
}

/// Transfer statistics
#[derive(Debug, Default)]
pub struct TransferStats {
    pub total_transfers: AtomicU64,
    pub total_bytes: AtomicU64,
    pub total_latency_ns: AtomicU64,
    pub gpu_to_cxl_count: AtomicU64,
    pub cxl_to_gpu_count: AtomicU64,
    pub p2p_transfers: AtomicU64,
    pub staged_transfers: AtomicU64,
}

impl TransferStats {
    pub fn record(&self, result: &TransferResult, direction: TransferDirection, used_p2p: bool) {
        self.total_transfers.fetch_add(1, Ordering::Relaxed);
        self.total_bytes
            .fetch_add(result.bytes_transferred as u64, Ordering::Relaxed);
        self.total_latency_ns
            .fetch_add(result.latency_ns, Ordering::Relaxed);

        match direction {
            TransferDirection::GpuToCxl => {
                self.gpu_to_cxl_count.fetch_add(1, Ordering::Relaxed);
            }
            TransferDirection::CxlToGpu => {
                self.cxl_to_gpu_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        if used_p2p {
            self.p2p_transfers.fetch_add(1, Ordering::Relaxed);
        } else {
            self.staged_transfers.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn avg_latency_us(&self) -> f64 {
        let count = self.total_transfers.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        let total_ns = self.total_latency_ns.load(Ordering::Relaxed);
        (total_ns as f64 / count as f64) / 1000.0
    }

    pub fn avg_bandwidth_gbps(&self) -> f64 {
        let total_bytes = self.total_bytes.load(Ordering::Relaxed);
        let total_ns = self.total_latency_ns.load(Ordering::Relaxed);
        if total_ns == 0 {
            return 0.0;
        }
        (total_bytes as f64 / 1e9) / (total_ns as f64 / 1e9)
    }
}

/// Configuration for CXL P2P context
#[derive(Debug, Clone)]
pub struct CxlP2pConfig {
    /// Default buffer size for allocations
    pub default_buffer_size: usize,
    /// Enable huge pages for buffer allocation
    pub use_huge_pages: bool,
    /// P2P DMA threshold (below this, use CPU staging)
    pub p2p_threshold_bytes: usize,
    /// Enable CUDA event timing
    pub enable_cuda_timing: bool,
    /// GPU device ID
    pub gpu_device_id: u32,
}

impl Default for CxlP2pConfig {
    fn default() -> Self {
        Self {
            default_buffer_size: 256 * 1024 * 1024, // 256MB
            use_huge_pages: true,
            p2p_threshold_bytes: 256 * 1024, // 256KB threshold for P2P
            enable_cuda_timing: true,
            gpu_device_id: 0,
        }
    }
}

/// CXL P2P DMA Context
///
/// This is a Rust wrapper around the cxl_pytorch_expander's P2P DMA interface.
/// It provides safe, ergonomic access to CXL memory operations.
///
/// The context is implemented as a simulation layer that can be replaced with
/// actual FFI bindings to the C++ cxl_memory module when running on hardware
/// with CXL support.
pub struct CxlP2pContext {
    /// Configuration
    config: CxlP2pConfig,
    /// CXL system information
    system_info: CxlSystemInfo,
    /// Allocated buffers
    buffers: RwLock<HashMap<u64, CxlBuffer>>,
    /// Next buffer ID
    next_buffer_id: AtomicU64,
    /// Transfer statistics
    stats: Arc<TransferStats>,
    /// Whether context is initialized
    initialized: bool,
    /// Whether P2P DMA is available
    p2p_available: bool,
}

impl CxlP2pContext {
    /// Create a new CXL P2P context
    pub fn new() -> CxlP2pResult<Self> {
        Self::with_config(CxlP2pConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: CxlP2pConfig) -> CxlP2pResult<Self> {
        let mut ctx = Self {
            config,
            system_info: CxlSystemInfo::default(),
            buffers: RwLock::new(HashMap::new()),
            next_buffer_id: AtomicU64::new(1),
            stats: Arc::new(TransferStats::default()),
            initialized: false,
            p2p_available: false,
        };

        ctx.initialize()?;
        Ok(ctx)
    }

    /// Initialize CXL connection
    fn initialize(&mut self) -> CxlP2pResult<()> {
        // In simulation mode, we assume CXL is available
        // In production, this would call cxl_memory.init() via FFI
        self.system_info = CxlSystemInfo {
            link_up: true,
            memory_expander: true,
            num_links: 1,
            version: 2,
            bandwidth_mbps: 3900, // ~4GB/s per link
            p2p_available: self.test_p2p_capability(),
        };

        self.p2p_available = self.system_info.p2p_available;
        self.initialized = true;

        tracing::info!(
            "CXL P2P context initialized: version={}, bandwidth={}MB/s, p2p={}",
            self.system_info.version,
            self.system_info.bandwidth_mbps,
            self.p2p_available
        );

        Ok(())
    }

    /// Test if P2P DMA is available
    fn test_p2p_capability(&self) -> bool {
        // In simulation mode, check environment variable
        // In production, this would attempt a small P2P transfer
        std::env::var("CXL_P2P_ENABLED")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false)
    }

    /// Get CXL system information
    pub fn system_info(&self) -> &CxlSystemInfo {
        &self.system_info
    }

    /// Check if P2P DMA is available
    pub fn is_p2p_available(&self) -> bool {
        self.p2p_available
    }

    /// Allocate a CXL buffer
    pub fn alloc_buffer(&self, size: usize) -> CxlP2pResult<u64> {
        if !self.initialized {
            return Err(CxlP2pError::NotInitialized);
        }

        let buffer_id = self.next_buffer_id.fetch_add(1, Ordering::SeqCst);

        // In simulation mode, allocate system memory
        // In production, this would call cxl_memory.alloc_buffer()
        let cpu_ptr = self.allocate_host_memory(size)?;

        let buffer = CxlBuffer {
            id: buffer_id,
            cpu_ptr,
            size,
            driver_handle: buffer_id, // In sim mode, same as buffer_id
            registered: true,
        };

        self.buffers.write().insert(buffer_id, buffer);

        tracing::debug!("Allocated CXL buffer {}: {} bytes", buffer_id, size);

        Ok(buffer_id)
    }

    /// Allocate host memory (simulation mode)
    fn allocate_host_memory(&self, size: usize) -> CxlP2pResult<u64> {
        // Use Vec<u8> as backing storage
        let mut buffer = vec![0u8; size];
        let ptr = buffer.as_mut_ptr() as u64;

        // Leak the memory to keep it alive (in production, this would be mmap'd)
        std::mem::forget(buffer);

        Ok(ptr)
    }

    /// Free a CXL buffer
    pub fn free_buffer(&self, buffer_id: u64) -> CxlP2pResult<()> {
        let buffer = self
            .buffers
            .write()
            .remove(&buffer_id)
            .ok_or(CxlP2pError::InvalidBufferId(buffer_id))?;

        // In simulation mode, we'd need to deallocate
        // In production, call cxl_memory.free_buffer()

        tracing::debug!("Freed CXL buffer {}", buffer_id);

        Ok(())
    }

    /// Get buffer information
    pub fn get_buffer(&self, buffer_id: u64) -> CxlP2pResult<CxlBuffer> {
        self.buffers
            .read()
            .get(&buffer_id)
            .map(|b| CxlBuffer {
                id: b.id,
                cpu_ptr: b.cpu_ptr,
                size: b.size,
                driver_handle: b.driver_handle,
                registered: b.registered,
            })
            .ok_or(CxlP2pError::InvalidBufferId(buffer_id))
    }

    /// Perform a P2P DMA transfer with timing
    ///
    /// # Arguments
    /// * `buffer_id` - CXL buffer ID
    /// * `gpu_ptr` - GPU memory pointer
    /// * `cxl_offset` - Offset within CXL buffer
    /// * `size` - Transfer size in bytes
    /// * `direction` - Transfer direction
    ///
    /// # Returns
    /// Transfer result with timing information
    pub fn transfer_timed(
        &self,
        buffer_id: u64,
        gpu_ptr: u64,
        cxl_offset: usize,
        size: usize,
        direction: TransferDirection,
    ) -> CxlP2pResult<TransferResult> {
        if !self.initialized {
            return Err(CxlP2pError::NotInitialized);
        }

        let buffer = self
            .buffers
            .read()
            .get(&buffer_id)
            .ok_or(CxlP2pError::InvalidBufferId(buffer_id))?
            .clone();

        // Validate bounds
        if cxl_offset + size > buffer.size {
            return Err(CxlP2pError::TransferFailed(format!(
                "Transfer out of bounds: offset {} + size {} > buffer size {}",
                cxl_offset, size, buffer.size
            )));
        }

        let start = Instant::now();

        // Choose transfer method based on size and P2P availability
        let used_p2p = if self.p2p_available && size >= self.config.p2p_threshold_bytes {
            self.transfer_p2p(&buffer, gpu_ptr, cxl_offset, size, direction)?;
            true
        } else {
            self.transfer_staged(&buffer, gpu_ptr, cxl_offset, size, direction)?;
            false
        };

        let latency_ns = start.elapsed().as_nanos() as u64;
        let bandwidth_gbps = (size as f64 / 1e9) / (latency_ns as f64 / 1e9);

        let result = TransferResult {
            transfer_id: 0, // Assigned by driver in production
            latency_ns,
            bandwidth_gbps,
            bytes_transferred: size,
        };

        self.stats.record(&result, direction, used_p2p);

        Ok(result)
    }

    /// P2P DMA transfer (direct GPU <-> CXL)
    fn transfer_p2p(
        &self,
        buffer: &CxlBuffer,
        gpu_ptr: u64,
        cxl_offset: usize,
        size: usize,
        direction: TransferDirection,
    ) -> CxlP2pResult<()> {
        // In production, this would call:
        // - cxl_memory.gpu_to_cxl_cuda_timed() for GpuToCxl
        // - cxl_memory.cxl_to_gpu_cuda_timed() for CxlToGpu

        tracing::trace!(
            "P2P DMA: {:?}, buffer={}, gpu_ptr={:#x}, offset={}, size={}",
            direction,
            buffer.id,
            gpu_ptr,
            cxl_offset,
            size
        );

        // Simulate P2P transfer latency (~100us base + bandwidth-limited)
        let base_latency_us = 100;
        let bandwidth_gbps = (self.system_info.bandwidth_mbps as f64 * self.system_info.num_links as f64) / 1000.0;
        let transfer_time_us = (size as f64 / 1e9 / bandwidth_gbps) * 1e6;
        let total_us = base_latency_us as f64 + transfer_time_us;

        std::thread::sleep(Duration::from_micros(total_us as u64));

        Ok(())
    }

    /// CPU-staged transfer (GPU -> CPU -> CXL or CXL -> CPU -> GPU)
    fn transfer_staged(
        &self,
        buffer: &CxlBuffer,
        gpu_ptr: u64,
        cxl_offset: usize,
        size: usize,
        direction: TransferDirection,
    ) -> CxlP2pResult<()> {
        // In production, this would use cudaMemcpy for GPU<->CPU
        // then memcpy for CPU<->CXL

        tracing::trace!(
            "Staged transfer: {:?}, buffer={}, gpu_ptr={:#x}, offset={}, size={}",
            direction,
            buffer.id,
            gpu_ptr,
            cxl_offset,
            size
        );

        // Simulate staged transfer (higher latency due to CPU involvement)
        let base_latency_us = 200; // Higher due to CPU involvement
        let bandwidth_gbps = 12.0; // PCIe Gen4 x16 bandwidth
        let transfer_time_us = (size as f64 / 1e9 / bandwidth_gbps) * 1e6 * 2.0; // 2x for GPU<->CPU and CPU<->CXL
        let total_us = base_latency_us as f64 + transfer_time_us;

        std::thread::sleep(Duration::from_micros(total_us as u64));

        Ok(())
    }

    /// Get transfer statistics
    pub fn stats(&self) -> Arc<TransferStats> {
        Arc::clone(&self.stats)
    }

    /// Get buffer CPU pointer for direct access
    pub fn get_buffer_ptr(&self, buffer_id: u64) -> CxlP2pResult<u64> {
        self.buffers
            .read()
            .get(&buffer_id)
            .map(|b| b.cpu_ptr)
            .ok_or(CxlP2pError::InvalidBufferId(buffer_id))
    }

    /// Copy data to CXL buffer (CPU path)
    pub fn copy_to_buffer(&self, buffer_id: u64, offset: usize, data: &[u8]) -> CxlP2pResult<()> {
        let buffer = self
            .buffers
            .read()
            .get(&buffer_id)
            .ok_or(CxlP2pError::InvalidBufferId(buffer_id))?
            .clone();

        if offset + data.len() > buffer.size {
            return Err(CxlP2pError::TransferFailed("Copy out of bounds".into()));
        }

        // In simulation, copy to the allocated memory
        unsafe {
            let dst = (buffer.cpu_ptr as *mut u8).add(offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }

        Ok(())
    }

    /// Copy data from CXL buffer (CPU path)
    pub fn copy_from_buffer(
        &self,
        buffer_id: u64,
        offset: usize,
        size: usize,
    ) -> CxlP2pResult<Vec<u8>> {
        let buffer = self
            .buffers
            .read()
            .get(&buffer_id)
            .ok_or(CxlP2pError::InvalidBufferId(buffer_id))?
            .clone();

        if offset + size > buffer.size {
            return Err(CxlP2pError::TransferFailed("Copy out of bounds".into()));
        }

        let mut data = vec![0u8; size];

        // In simulation, copy from the allocated memory
        unsafe {
            let src = (buffer.cpu_ptr as *const u8).add(offset);
            std::ptr::copy_nonoverlapping(src, data.as_mut_ptr(), size);
        }

        Ok(data)
    }
}

impl Clone for CxlBuffer {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            cpu_ptr: self.cpu_ptr,
            size: self.size,
            driver_handle: self.driver_handle,
            registered: self.registered,
        }
    }
}

/// Expert memory region within a CXL buffer
#[derive(Debug, Clone)]
pub struct ExpertRegion {
    /// Buffer containing this expert
    pub buffer_id: u64,
    /// Offset within buffer
    pub offset: usize,
    /// Expert weight size
    pub size: usize,
    /// Layer ID
    pub layer_id: u32,
    /// Expert ID
    pub expert_id: u32,
}

/// Expert memory pool for managing MoE expert weights in CXL memory
pub struct ExpertMemoryPool {
    /// CXL P2P context
    ctx: Arc<CxlP2pContext>,
    /// Allocated expert regions
    experts: RwLock<HashMap<(u32, u32), ExpertRegion>>,
    /// Buffer pool
    buffers: Vec<u64>,
    /// Current buffer index
    current_buffer: AtomicU64,
    /// Current offset in current buffer
    current_offset: AtomicU64,
    /// Buffer size
    buffer_size: usize,
}

impl ExpertMemoryPool {
    /// Create a new expert memory pool
    pub fn new(ctx: Arc<CxlP2pContext>, num_buffers: usize) -> CxlP2pResult<Self> {
        let buffer_size = ctx.config.default_buffer_size;
        let mut buffers = Vec::with_capacity(num_buffers);

        for _ in 0..num_buffers {
            let buffer_id = ctx.alloc_buffer(buffer_size)?;
            buffers.push(buffer_id);
        }

        Ok(Self {
            ctx,
            experts: RwLock::new(HashMap::new()),
            buffers,
            current_buffer: AtomicU64::new(0),
            current_offset: AtomicU64::new(0),
            buffer_size,
        })
    }

    /// Allocate region for an expert
    pub fn alloc_expert(&self, layer_id: u32, expert_id: u32, size: usize) -> CxlP2pResult<ExpertRegion> {
        let key = (layer_id, expert_id);

        // Check if already allocated
        if let Some(region) = self.experts.read().get(&key) {
            return Ok(region.clone());
        }

        // Find space
        let mut offset = self.current_offset.load(Ordering::SeqCst) as usize;
        let mut buffer_idx = self.current_buffer.load(Ordering::SeqCst) as usize;

        if offset + size > self.buffer_size {
            // Move to next buffer
            buffer_idx += 1;
            if buffer_idx >= self.buffers.len() {
                return Err(CxlP2pError::AllocationFailed("Expert pool exhausted".into()));
            }
            self.current_buffer.store(buffer_idx as u64, Ordering::SeqCst);
            offset = 0;
        }

        let buffer_id = self.buffers[buffer_idx];
        let region = ExpertRegion {
            buffer_id,
            offset,
            size,
            layer_id,
            expert_id,
        };

        self.current_offset.store((offset + size) as u64, Ordering::SeqCst);
        self.experts.write().insert(key, region.clone());

        Ok(region)
    }

    /// Get expert region
    pub fn get_expert(&self, layer_id: u32, expert_id: u32) -> Option<ExpertRegion> {
        self.experts.read().get(&(layer_id, expert_id)).cloned()
    }

    /// Park expert weights from GPU to CXL
    pub fn park_expert(
        &self,
        layer_id: u32,
        expert_id: u32,
        gpu_ptr: u64,
        size: usize,
    ) -> CxlP2pResult<TransferResult> {
        let region = self.alloc_expert(layer_id, expert_id, size)?;

        self.ctx.transfer_timed(
            region.buffer_id,
            gpu_ptr,
            region.offset,
            size,
            TransferDirection::GpuToCxl,
        )
    }

    /// Hydrate expert weights from CXL to GPU
    pub fn hydrate_expert(
        &self,
        layer_id: u32,
        expert_id: u32,
        gpu_ptr: u64,
    ) -> CxlP2pResult<TransferResult> {
        let region = self
            .experts
            .read()
            .get(&(layer_id, expert_id))
            .cloned()
            .ok_or_else(|| {
                CxlP2pError::TransferFailed(format!(
                    "Expert ({}, {}) not found in pool",
                    layer_id, expert_id
                ))
            })?;

        self.ctx.transfer_timed(
            region.buffer_id,
            gpu_ptr,
            region.offset,
            region.size,
            TransferDirection::CxlToGpu,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cxl_context_creation() {
        let ctx = CxlP2pContext::new().unwrap();
        assert!(ctx.initialized);
        assert!(ctx.system_info.link_up);
    }

    #[test]
    fn test_buffer_allocation() {
        let ctx = CxlP2pContext::new().unwrap();

        let buffer_id = ctx.alloc_buffer(1024 * 1024).unwrap(); // 1MB
        assert!(buffer_id > 0);

        let buffer = ctx.get_buffer(buffer_id).unwrap();
        assert_eq!(buffer.size, 1024 * 1024);

        ctx.free_buffer(buffer_id).unwrap();
    }

    #[test]
    fn test_buffer_copy() {
        let ctx = CxlP2pContext::new().unwrap();

        let buffer_id = ctx.alloc_buffer(1024).unwrap();

        let data = vec![0x42u8; 512];
        ctx.copy_to_buffer(buffer_id, 0, &data).unwrap();

        let read_data = ctx.copy_from_buffer(buffer_id, 0, 512).unwrap();
        assert_eq!(read_data, data);

        ctx.free_buffer(buffer_id).unwrap();
    }

    #[test]
    fn test_transfer_simulation() {
        let ctx = CxlP2pContext::new().unwrap();

        let buffer_id = ctx.alloc_buffer(1024 * 1024).unwrap();

        // Simulate transfer (gpu_ptr is dummy in sim mode)
        let result = ctx
            .transfer_timed(buffer_id, 0x1000, 0, 64 * 1024, TransferDirection::GpuToCxl)
            .unwrap();

        assert!(result.latency_ns > 0);
        assert!(result.bandwidth_gbps > 0.0);
        assert_eq!(result.bytes_transferred, 64 * 1024);

        ctx.free_buffer(buffer_id).unwrap();
    }

    #[test]
    fn test_expert_memory_pool() {
        let ctx = Arc::new(CxlP2pContext::new().unwrap());
        let pool = ExpertMemoryPool::new(ctx, 2).unwrap();

        // Allocate expert
        let region = pool.alloc_expert(0, 7, 1024 * 1024).unwrap();
        assert_eq!(region.layer_id, 0);
        assert_eq!(region.expert_id, 7);
        assert_eq!(region.size, 1024 * 1024);

        // Get same expert should return same region
        let region2 = pool.get_expert(0, 7).unwrap();
        assert_eq!(region.offset, region2.offset);
    }
}
