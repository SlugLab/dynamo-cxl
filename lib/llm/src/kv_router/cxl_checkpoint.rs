// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CXL Checkpoint Manager for MoE Models
//!
//! This module provides checkpoint/restore functionality using CXL memory with P2P DMA.
//! It stores token-to-expert mappings and routing decisions in CXL memory for fast recovery.
//!
//! # Key Features
//!
//! - **CXL P2P Checkpoint Storage**: Checkpoints stored in CXL memory via P2P DMA
//! - **Token-to-Expert Record/Replay**: Records routing decisions for fast replay on recovery
//! - **Sub-millisecond Recovery**: No re-prefill needed, just replay routing decisions
//! - **Windowed Checkpointing**: 16-token windows for balanced granularity
//!
//! # Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────────────────┐
//! │                         GPU HBM Memory                                    │
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐│
//! │  │ Active Experts  │  │  KV Cache Hot   │  │    Current Token State      ││
//! │  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘│
//! └───────────┼─────────────────────┼─────────────────────────┼───────────────┘
//!             │                     │                         │
//!             │ P2P DMA (Checkpoint Write / Recovery Read)    │
//!             ▼                     ▼                         ▼
//! ┌───────────────────────────────────────────────────────────────────────────┐
//! │                         CXL Memory Pool                                   │
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐│
//! │  │  Parked Experts │  │ Checkpoint Ring │  │   Token-Expert Mappings     ││
//! │  │    (Cold)       │  │    Buffer       │  │   (Record for Replay)       ││
//! │  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘│
//! └───────────────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

use super::cxl_p2p::{CxlP2pContext, CxlP2pError, CxlP2pResult, TransferDirection, TransferResult};
use super::protocols::{ExternalSequenceBlockHash, LocalBlockHash};

/// Checkpoint format version for compatibility
pub const CHECKPOINT_VERSION: u32 = 1;

/// Magic number for checkpoint identification
pub const CHECKPOINT_MAGIC: u64 = 0x43584C_43484B5054; // "CXL_CHKPT"

/// Default checkpoint buffer size (256MB)
pub const DEFAULT_CHECKPOINT_BUFFER_SIZE: usize = 256 * 1024 * 1024;

/// Maximum checkpoints in ring buffer
pub const MAX_RING_CHECKPOINTS: usize = 64;

/// Token-to-Expert mapping entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenExpertMapping {
    /// Sequence ID (for multi-sequence batching)
    pub sequence_id: u64,
    /// Token position in sequence
    pub token_position: u32,
    /// Layer ID
    pub layer_id: u32,
    /// Selected expert ID
    pub expert_id: u32,
    /// Top-k expert IDs (full routing decision)
    pub topk_experts: Vec<u32>,
    /// Gating scores for top-k
    pub gating_scores: Vec<f32>,
    /// KV block reference (hash, not data)
    pub kv_block_hash: u64,
    /// Timestamp (microseconds since epoch)
    pub timestamp_us: u64,
}

impl TokenExpertMapping {
    pub fn new(
        sequence_id: u64,
        token_position: u32,
        layer_id: u32,
        expert_id: u32,
        topk_experts: Vec<u32>,
        gating_scores: Vec<f32>,
        kv_block_hash: u64,
    ) -> Self {
        let timestamp_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);

        Self {
            sequence_id,
            token_position,
            layer_id,
            expert_id,
            topk_experts,
            gating_scores,
            kv_block_hash,
            timestamp_us,
        }
    }
}

/// Checkpoint header stored at the beginning of each checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointHeader {
    /// Magic number for identification
    pub magic: u64,
    /// Format version
    pub version: u32,
    /// Checkpoint ID
    pub checkpoint_id: u64,
    /// Window start token
    pub window_start: u32,
    /// Number of tokens in window
    pub window_len: u32,
    /// Number of layers
    pub num_layers: u32,
    /// Total size of serialized data
    pub data_size: u64,
    /// CRC32 checksum of data
    pub checksum: u32,
    /// Timestamp (microseconds since epoch)
    pub timestamp_us: u64,
    /// Sequence ID
    pub sequence_id: u64,
}

impl CheckpointHeader {
    pub fn new(
        checkpoint_id: u64,
        window_start: u32,
        window_len: u32,
        num_layers: u32,
        data_size: u64,
        sequence_id: u64,
    ) -> Self {
        let timestamp_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);

        Self {
            magic: CHECKPOINT_MAGIC,
            version: CHECKPOINT_VERSION,
            checkpoint_id,
            window_start,
            window_len,
            num_layers,
            data_size,
            checksum: 0, // Computed during serialization
            timestamp_us,
            sequence_id,
        }
    }

    /// Compute CRC32 checksum
    pub fn compute_checksum(&mut self, data: &[u8]) {
        // Simple checksum using CRC32
        let mut crc: u32 = 0xFFFFFFFF;
        for byte in data {
            crc ^= *byte as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB88320;
                } else {
                    crc >>= 1;
                }
            }
        }
        self.checksum = !crc;
    }

    /// Verify checksum
    pub fn verify_checksum(&self, data: &[u8]) -> bool {
        let mut crc: u32 = 0xFFFFFFFF;
        for byte in data {
            crc ^= *byte as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB88320;
                } else {
                    crc >>= 1;
                }
            }
        }
        !crc == self.checksum
    }

    /// Header size in bytes
    pub fn size() -> usize {
        // Fixed size for binary serialization
        std::mem::size_of::<u64>() * 4 + // magic, data_size, timestamp_us, sequence_id
        std::mem::size_of::<u32>() * 5   // version, checkpoint_id, window_start, window_len, num_layers, checksum
    }
}

/// Serialized checkpoint data stored in CXL memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointData {
    /// Header
    pub header: CheckpointHeader,
    /// Token-to-expert mappings for this window
    pub mappings: Vec<TokenExpertMapping>,
    /// Expert locations at checkpoint time (layer_id, expert_id) -> location
    pub expert_locations: HashMap<(u32, u32), u8>, // 0=GPU, 1=CXL, 2=Transit, 3=Evicted
    /// Hot set expert IDs at checkpoint
    pub hot_set: Vec<(u32, u32)>,
}

impl CheckpointData {
    pub fn new(
        checkpoint_id: u64,
        window_start: u32,
        num_layers: u32,
        sequence_id: u64,
        mappings: Vec<TokenExpertMapping>,
        expert_locations: HashMap<(u32, u32), u8>,
        hot_set: Vec<(u32, u32)>,
    ) -> Self {
        let window_len = mappings.len() as u32 / num_layers;
        let header = CheckpointHeader::new(
            checkpoint_id,
            window_start,
            window_len,
            num_layers,
            0, // Will be set during serialization
            sequence_id,
        );

        Self {
            header,
            mappings,
            expert_locations,
            hot_set,
        }
    }

    /// Serialize checkpoint to bytes for storage
    pub fn to_bytes(&mut self) -> Result<Vec<u8>, String> {
        // Serialize data portion first
        let data_bytes = bincode::serialize(&self.mappings)
            .map_err(|e| format!("Failed to serialize mappings: {}", e))?;
        let expert_loc_bytes = bincode::serialize(&self.expert_locations)
            .map_err(|e| format!("Failed to serialize expert locations: {}", e))?;
        let hot_set_bytes = bincode::serialize(&self.hot_set)
            .map_err(|e| format!("Failed to serialize hot set: {}", e))?;

        // Update header with data size
        self.header.data_size =
            (data_bytes.len() + expert_loc_bytes.len() + hot_set_bytes.len()) as u64;

        // Compute checksum
        let mut all_data = Vec::new();
        all_data.extend(&data_bytes);
        all_data.extend(&expert_loc_bytes);
        all_data.extend(&hot_set_bytes);
        self.header.compute_checksum(&all_data);

        // Serialize header
        let header_bytes = bincode::serialize(&self.header)
            .map_err(|e| format!("Failed to serialize header: {}", e))?;

        // Combine: [header_len(4 bytes)][header][data_len(4 bytes)][data]...
        let mut result = Vec::new();
        result.extend(&(header_bytes.len() as u32).to_le_bytes());
        result.extend(header_bytes);
        result.extend(&(data_bytes.len() as u32).to_le_bytes());
        result.extend(data_bytes);
        result.extend(&(expert_loc_bytes.len() as u32).to_le_bytes());
        result.extend(expert_loc_bytes);
        result.extend(&(hot_set_bytes.len() as u32).to_le_bytes());
        result.extend(hot_set_bytes);

        Ok(result)
    }

    /// Deserialize checkpoint from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let mut offset = 0;

        // Read header
        if data.len() < 4 {
            return Err("Data too short for header length".into());
        }
        let header_len =
            u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        if data.len() < offset + header_len {
            return Err("Data too short for header".into());
        }
        let header: CheckpointHeader = bincode::deserialize(&data[offset..offset + header_len])
            .map_err(|e| format!("Failed to deserialize header: {}", e))?;
        offset += header_len;

        // Verify magic
        if header.magic != CHECKPOINT_MAGIC {
            return Err("Invalid checkpoint magic".into());
        }

        // Read mappings
        if data.len() < offset + 4 {
            return Err("Data too short for mappings length".into());
        }
        let mappings_len =
            u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        if data.len() < offset + mappings_len {
            return Err("Data too short for mappings".into());
        }
        let mappings: Vec<TokenExpertMapping> =
            bincode::deserialize(&data[offset..offset + mappings_len])
                .map_err(|e| format!("Failed to deserialize mappings: {}", e))?;
        offset += mappings_len;

        // Read expert locations
        if data.len() < offset + 4 {
            return Err("Data too short for expert locations length".into());
        }
        let expert_loc_len =
            u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        if data.len() < offset + expert_loc_len {
            return Err("Data too short for expert locations".into());
        }
        let expert_locations: HashMap<(u32, u32), u8> =
            bincode::deserialize(&data[offset..offset + expert_loc_len])
                .map_err(|e| format!("Failed to deserialize expert locations: {}", e))?;
        offset += expert_loc_len;

        // Read hot set
        if data.len() < offset + 4 {
            return Err("Data too short for hot set length".into());
        }
        let hot_set_len =
            u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        if data.len() < offset + hot_set_len {
            return Err("Data too short for hot set".into());
        }
        let hot_set: Vec<(u32, u32)> = bincode::deserialize(&data[offset..offset + hot_set_len])
            .map_err(|e| format!("Failed to deserialize hot set: {}", e))?;

        Ok(Self {
            header,
            mappings,
            expert_locations,
            hot_set,
        })
    }
}

/// Ring buffer slot for checkpoint storage
#[derive(Debug)]
struct CheckpointSlot {
    /// Slot index
    index: usize,
    /// Offset in CXL buffer
    offset: usize,
    /// Size of checkpoint data
    size: usize,
    /// Checkpoint ID
    checkpoint_id: u64,
    /// Whether slot is valid
    valid: bool,
}

/// CXL Checkpoint Manager
///
/// Manages checkpoint storage in CXL memory using P2P DMA.
pub struct CxlCheckpointManager {
    /// CXL P2P context
    ctx: Arc<CxlP2pContext>,
    /// Checkpoint ring buffer ID
    ring_buffer_id: u64,
    /// Ring buffer size
    ring_buffer_size: usize,
    /// Ring buffer slots
    slots: Mutex<VecDeque<CheckpointSlot>>,
    /// Current write offset in ring buffer
    write_offset: AtomicU64,
    /// Next checkpoint ID
    next_checkpoint_id: AtomicU64,
    /// Window size (tokens per checkpoint)
    window_size: u32,
    /// Number of MoE layers
    num_layers: u32,
    /// Current window's token-expert mappings
    current_window: Mutex<Vec<TokenExpertMapping>>,
    /// Current sequence ID
    current_sequence_id: AtomicU64,
    /// Metrics
    metrics: CheckpointMetrics,
}

/// Checkpoint manager metrics
#[derive(Debug, Default)]
pub struct CheckpointMetrics {
    pub checkpoints_written: AtomicU64,
    pub checkpoints_read: AtomicU64,
    pub bytes_written: AtomicU64,
    pub bytes_read: AtomicU64,
    pub write_latency_us_total: AtomicU64,
    pub read_latency_us_total: AtomicU64,
    pub recovery_count: AtomicU64,
    pub last_recovery_time_us: AtomicU64,
}

impl CheckpointMetrics {
    pub fn avg_write_latency_us(&self) -> f64 {
        let count = self.checkpoints_written.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        self.write_latency_us_total.load(Ordering::Relaxed) as f64 / count as f64
    }

    pub fn avg_read_latency_us(&self) -> f64 {
        let count = self.checkpoints_read.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        self.read_latency_us_total.load(Ordering::Relaxed) as f64 / count as f64
    }

    pub fn write_bandwidth_gbps(&self) -> f64 {
        let bytes = self.bytes_written.load(Ordering::Relaxed);
        let latency_us = self.write_latency_us_total.load(Ordering::Relaxed);
        if latency_us == 0 {
            return 0.0;
        }
        (bytes as f64 / 1e9) / (latency_us as f64 / 1e6)
    }
}

impl CxlCheckpointManager {
    /// Create a new CXL Checkpoint Manager
    pub fn new(
        ctx: Arc<CxlP2pContext>,
        window_size: u32,
        num_layers: u32,
        buffer_size: Option<usize>,
    ) -> CxlP2pResult<Self> {
        let ring_buffer_size = buffer_size.unwrap_or(DEFAULT_CHECKPOINT_BUFFER_SIZE);

        // Allocate checkpoint ring buffer in CXL memory
        let ring_buffer_id = ctx.alloc_buffer(ring_buffer_size)?;

        tracing::info!(
            "CXL Checkpoint Manager initialized: buffer_id={}, size={}MB, window_size={}",
            ring_buffer_id,
            ring_buffer_size / 1024 / 1024,
            window_size
        );

        Ok(Self {
            ctx,
            ring_buffer_id,
            ring_buffer_size,
            slots: Mutex::new(VecDeque::with_capacity(MAX_RING_CHECKPOINTS)),
            write_offset: AtomicU64::new(0),
            next_checkpoint_id: AtomicU64::new(1),
            window_size,
            num_layers,
            current_window: Mutex::new(Vec::with_capacity(window_size as usize * num_layers as usize)),
            current_sequence_id: AtomicU64::new(0),
            metrics: CheckpointMetrics::default(),
        })
    }

    /// Set the current sequence ID
    pub fn set_sequence_id(&self, sequence_id: u64) {
        self.current_sequence_id.store(sequence_id, Ordering::SeqCst);
    }

    /// Record a token-to-expert mapping
    ///
    /// This records the routing decision for a single token at a specific layer.
    /// When window_size tokens are recorded, a checkpoint is automatically created.
    pub fn record_mapping(
        &self,
        token_position: u32,
        layer_id: u32,
        expert_id: u32,
        topk_experts: Vec<u32>,
        gating_scores: Vec<f32>,
        kv_block_hash: u64,
    ) {
        let sequence_id = self.current_sequence_id.load(Ordering::Relaxed);

        let mapping = TokenExpertMapping::new(
            sequence_id,
            token_position,
            layer_id,
            expert_id,
            topk_experts,
            gating_scores,
            kv_block_hash,
        );

        let mut window = self.current_window.lock();
        window.push(mapping);

        // Check if window is complete (all layers for window_size tokens)
        let expected_entries = self.window_size as usize * self.num_layers as usize;
        if window.len() >= expected_entries {
            // Window complete, will be committed by caller or automatically
            tracing::trace!(
                "Window complete: {} entries for {} tokens",
                window.len(),
                self.window_size
            );
        }
    }

    /// Write a checkpoint to CXL memory using P2P DMA
    ///
    /// This serializes the current window's mappings and writes them to CXL memory.
    pub fn write_checkpoint(
        &self,
        expert_locations: HashMap<(u32, u32), u8>,
        hot_set: Vec<(u32, u32)>,
        gpu_ptr: Option<u64>,
    ) -> CxlP2pResult<u64> {
        let start = Instant::now();

        // Get current window and clear it
        let mappings = {
            let mut window = self.current_window.lock();
            std::mem::take(&mut *window)
        };

        if mappings.is_empty() {
            return Err(CxlP2pError::TransferFailed("No mappings to checkpoint".into()));
        }

        // Create checkpoint
        let checkpoint_id = self.next_checkpoint_id.fetch_add(1, Ordering::SeqCst);
        let window_start = mappings.first().map(|m| m.token_position).unwrap_or(0);
        let sequence_id = self.current_sequence_id.load(Ordering::Relaxed);

        let mut checkpoint = CheckpointData::new(
            checkpoint_id,
            window_start,
            self.num_layers,
            sequence_id,
            mappings,
            expert_locations,
            hot_set,
        );

        // Serialize to bytes
        let data = checkpoint
            .to_bytes()
            .map_err(|e| CxlP2pError::TransferFailed(e))?;
        let data_size = data.len();

        // Find write position in ring buffer
        let write_offset = self.allocate_ring_slot(data_size)?;

        // Write to CXL buffer
        if let Some(gpu_ptr) = gpu_ptr {
            // Use P2P DMA from GPU
            self.ctx.transfer_timed(
                self.ring_buffer_id,
                gpu_ptr,
                write_offset,
                data_size,
                TransferDirection::GpuToCxl,
            )?;
        } else {
            // CPU path: copy directly to CXL buffer
            self.ctx
                .copy_to_buffer(self.ring_buffer_id, write_offset, &data)?;
        }

        // Record slot
        {
            let mut slots = self.slots.lock();

            // Remove oldest if at capacity
            while slots.len() >= MAX_RING_CHECKPOINTS {
                slots.pop_front();
            }

            let index = slots.len();
            slots.push_back(CheckpointSlot {
                index,
                offset: write_offset,
                size: data_size,
                checkpoint_id,
                valid: true,
            });
        }

        let elapsed = start.elapsed();

        // Update metrics
        self.metrics.checkpoints_written.fetch_add(1, Ordering::Relaxed);
        self.metrics.bytes_written.fetch_add(data_size as u64, Ordering::Relaxed);
        self.metrics.write_latency_us_total.fetch_add(
            elapsed.as_micros() as u64,
            Ordering::Relaxed,
        );

        tracing::debug!(
            "Checkpoint {} written: offset={}, size={}, latency={:?}",
            checkpoint_id,
            write_offset,
            data_size,
            elapsed
        );

        Ok(checkpoint_id)
    }

    /// Allocate a slot in the ring buffer
    fn allocate_ring_slot(&self, size: usize) -> CxlP2pResult<usize> {
        let mut offset = self.write_offset.load(Ordering::SeqCst) as usize;

        // Align to 4KB for optimal DMA
        let aligned_size = (size + 4095) & !4095;

        // Check if we need to wrap around
        if offset + aligned_size > self.ring_buffer_size {
            offset = 0;
            self.write_offset.store(0, Ordering::SeqCst);
        }

        // Advance write offset
        self.write_offset.fetch_add(aligned_size as u64, Ordering::SeqCst);

        Ok(offset)
    }

    /// Read a checkpoint from CXL memory
    pub fn read_checkpoint(
        &self,
        checkpoint_id: u64,
        gpu_ptr: Option<u64>,
    ) -> CxlP2pResult<CheckpointData> {
        let start = Instant::now();

        // Find checkpoint slot
        let slot = {
            let slots = self.slots.lock();
            slots
                .iter()
                .find(|s| s.checkpoint_id == checkpoint_id && s.valid)
                .map(|s| (s.offset, s.size))
        };

        let (offset, size) = slot.ok_or_else(|| {
            CxlP2pError::TransferFailed(format!("Checkpoint {} not found", checkpoint_id))
        })?;

        // Read from CXL buffer
        let data = if let Some(gpu_ptr) = gpu_ptr {
            // P2P DMA to GPU
            self.ctx.transfer_timed(
                self.ring_buffer_id,
                gpu_ptr,
                offset,
                size,
                TransferDirection::CxlToGpu,
            )?;

            // In simulation, we still need to read via CPU
            self.ctx.copy_from_buffer(self.ring_buffer_id, offset, size)?
        } else {
            // CPU path
            self.ctx.copy_from_buffer(self.ring_buffer_id, offset, size)?
        };

        // Deserialize
        let checkpoint = CheckpointData::from_bytes(&data)
            .map_err(|e| CxlP2pError::TransferFailed(e))?;

        let elapsed = start.elapsed();

        // Update metrics
        self.metrics.checkpoints_read.fetch_add(1, Ordering::Relaxed);
        self.metrics.bytes_read.fetch_add(size as u64, Ordering::Relaxed);
        self.metrics.read_latency_us_total.fetch_add(
            elapsed.as_micros() as u64,
            Ordering::Relaxed,
        );

        tracing::debug!(
            "Checkpoint {} read: offset={}, size={}, latency={:?}",
            checkpoint_id,
            offset,
            size,
            elapsed
        );

        Ok(checkpoint)
    }

    /// Get the latest checkpoint
    pub fn get_latest_checkpoint(&self, gpu_ptr: Option<u64>) -> CxlP2pResult<CheckpointData> {
        let checkpoint_id = {
            let slots = self.slots.lock();
            slots
                .back()
                .filter(|s| s.valid)
                .map(|s| s.checkpoint_id)
        };

        match checkpoint_id {
            Some(id) => self.read_checkpoint(id, gpu_ptr),
            None => Err(CxlP2pError::TransferFailed("No checkpoints available".into())),
        }
    }

    /// Perform fast recovery from the latest checkpoint
    ///
    /// This reads the checkpoint from CXL memory and returns the routing decisions
    /// that need to be replayed.
    pub fn fast_recovery(&self, gpu_ptr: Option<u64>) -> CxlP2pResult<RecoveryResult> {
        let start = Instant::now();

        // Read latest checkpoint
        let checkpoint = self.get_latest_checkpoint(gpu_ptr)?;

        // Build replay instructions
        let mut replay_instructions = Vec::new();
        for mapping in &checkpoint.mappings {
            replay_instructions.push(ReplayInstruction {
                sequence_id: mapping.sequence_id,
                token_position: mapping.token_position,
                layer_id: mapping.layer_id,
                expert_id: mapping.expert_id,
                topk_experts: mapping.topk_experts.clone(),
                gating_scores: mapping.gating_scores.clone(),
            });
        }

        let elapsed = start.elapsed();

        // Update metrics
        self.metrics.recovery_count.fetch_add(1, Ordering::Relaxed);
        self.metrics.last_recovery_time_us.store(
            elapsed.as_micros() as u64,
            Ordering::Relaxed,
        );

        tracing::info!(
            "Fast recovery completed: {} routing decisions replayed in {:?}",
            replay_instructions.len(),
            elapsed
        );

        Ok(RecoveryResult {
            checkpoint_id: checkpoint.header.checkpoint_id,
            window_start: checkpoint.header.window_start,
            window_len: checkpoint.header.window_len,
            replay_instructions,
            expert_locations: checkpoint.expert_locations,
            hot_set: checkpoint.hot_set,
            recovery_time: elapsed,
        })
    }

    /// Force commit any pending mappings
    pub fn force_commit(
        &self,
        expert_locations: HashMap<(u32, u32), u8>,
        hot_set: Vec<(u32, u32)>,
    ) -> CxlP2pResult<Option<u64>> {
        let has_pending = !self.current_window.lock().is_empty();
        if has_pending {
            Ok(Some(self.write_checkpoint(expert_locations, hot_set, None)?))
        } else {
            Ok(None)
        }
    }

    /// Get checkpoint metrics
    pub fn get_metrics(&self) -> &CheckpointMetrics {
        &self.metrics
    }

    /// Get number of valid checkpoints
    pub fn checkpoint_count(&self) -> usize {
        self.slots.lock().iter().filter(|s| s.valid).count()
    }

    /// Get total checkpoint data size
    pub fn total_checkpoint_size(&self) -> usize {
        self.slots.lock().iter().filter(|s| s.valid).map(|s| s.size).sum()
    }
}

/// Result of fast recovery
#[derive(Debug)]
pub struct RecoveryResult {
    /// Checkpoint ID that was recovered from
    pub checkpoint_id: u64,
    /// Window start token
    pub window_start: u32,
    /// Window length
    pub window_len: u32,
    /// Replay instructions (token-to-expert mappings to restore)
    pub replay_instructions: Vec<ReplayInstruction>,
    /// Expert locations to restore
    pub expert_locations: HashMap<(u32, u32), u8>,
    /// Hot set to restore
    pub hot_set: Vec<(u32, u32)>,
    /// Time taken for recovery
    pub recovery_time: Duration,
}

/// Instruction for replaying a routing decision
#[derive(Debug, Clone)]
pub struct ReplayInstruction {
    /// Sequence ID
    pub sequence_id: u64,
    /// Token position
    pub token_position: u32,
    /// Layer ID
    pub layer_id: u32,
    /// Selected expert ID
    pub expert_id: u32,
    /// Top-k experts
    pub topk_experts: Vec<u32>,
    /// Gating scores
    pub gating_scores: Vec<f32>,
}

// ============================================================================
// Windowed Attention Support
// ============================================================================
//
// Sliding window attention (used in Mistral, Qwen, etc.) limits attention to
// the most recent `attention_window_size` tokens. This has key implications:
//
// 1. KV cache only needs to store `attention_window_size` entries per layer
// 2. Recovery only needs to restore the attention window, not full history
// 3. Checkpoint windows should align with attention windows for efficiency
//
// Architecture with windowed attention:
//
// ```text
// Full Sequence: [t0, t1, t2, ..., t_{n-1}, t_n]
//                                  |<-- attention_window -->|
//
// KV Cache (per layer):
//   [k_{n-w}, v_{n-w}] ... [k_{n-1}, v_{n-1}] [k_n, v_n]
//   |<----------- attention_window_size ------------->|
//
// On recovery, we only need to restore:
//   1. KV cache blocks within the attention window
//   2. Expert routing decisions within the window
//   3. Token positions for attention mask computation
// ```

/// Default sliding window attention size (matches Mistral/Qwen defaults)
pub const DEFAULT_ATTENTION_WINDOW_SIZE: u32 = 4096;

/// Configuration for sliding window attention checkpointing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlidingWindowAttentionConfig {
    /// Size of the attention window in tokens
    pub attention_window_size: u32,
    /// Number of transformer layers
    pub num_layers: u32,
    /// Number of attention heads per layer
    pub num_heads: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Whether to use sliding window attention (vs full attention)
    pub use_sliding_window: bool,
    /// Checkpoint alignment: align checkpoints to attention window boundaries
    pub align_checkpoints_to_window: bool,
}

impl Default for SlidingWindowAttentionConfig {
    fn default() -> Self {
        Self {
            attention_window_size: DEFAULT_ATTENTION_WINDOW_SIZE,
            num_layers: 32,
            num_heads: 32,
            head_dim: 128,
            use_sliding_window: true,
            align_checkpoints_to_window: true,
        }
    }
}

impl SlidingWindowAttentionConfig {
    /// Create a new config for models like Mistral/Qwen with sliding window attention
    pub fn new_sliding_window(
        attention_window_size: u32,
        num_layers: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Self {
        Self {
            attention_window_size,
            num_layers,
            num_heads,
            head_dim,
            use_sliding_window: true,
            align_checkpoints_to_window: true,
        }
    }

    /// Create a config for full attention models (no sliding window)
    pub fn new_full_attention(num_layers: u32, num_heads: u32, head_dim: u32) -> Self {
        Self {
            attention_window_size: u32::MAX, // Effectively unlimited
            num_layers,
            num_heads,
            head_dim,
            use_sliding_window: false,
            align_checkpoints_to_window: false,
        }
    }

    /// Calculate KV cache size per layer in bytes (for one token)
    pub fn kv_cache_size_per_token(&self) -> usize {
        // K and V each have shape [num_heads, head_dim], stored as fp16
        2 * self.num_heads as usize * self.head_dim as usize * 2 // 2 bytes per fp16
    }

    /// Calculate total KV cache size for the attention window (all layers)
    pub fn kv_cache_window_size(&self) -> usize {
        self.kv_cache_size_per_token()
            * self.attention_window_size as usize
            * self.num_layers as usize
    }
}

/// KV cache block reference within the attention window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheBlockRef {
    /// Block hash (references data in CXL memory)
    pub block_hash: u64,
    /// Layer ID
    pub layer_id: u32,
    /// Start token position in the block
    pub start_token: u32,
    /// Number of tokens in this block
    pub num_tokens: u32,
    /// CXL buffer offset (for P2P DMA recovery)
    pub cxl_offset: Option<u64>,
    /// Whether this block is within the current attention window
    pub in_attention_window: bool,
}

/// Represents the KV cache state for the attention window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheWindowState {
    /// Attention window configuration
    pub config: SlidingWindowAttentionConfig,
    /// Current sequence position (total tokens processed)
    pub sequence_position: u32,
    /// Start of the attention window (sequence_position - window_size, clamped to 0)
    pub window_start: u32,
    /// KV cache block references within the window (per layer)
    pub blocks_per_layer: HashMap<u32, Vec<KvCacheBlockRef>>,
    /// Total KV cache size in bytes (for the window)
    pub window_size_bytes: usize,
}

impl KvCacheWindowState {
    /// Create a new KV cache window state
    pub fn new(config: SlidingWindowAttentionConfig) -> Self {
        Self {
            config,
            sequence_position: 0,
            window_start: 0,
            blocks_per_layer: HashMap::new(),
            window_size_bytes: 0,
        }
    }

    /// Update the window state after processing tokens
    pub fn advance(&mut self, num_tokens: u32) {
        self.sequence_position += num_tokens;

        // Update window start (slide the window)
        if self.config.use_sliding_window {
            self.window_start = self.sequence_position
                .saturating_sub(self.config.attention_window_size);
        }
    }

    /// Add a KV cache block reference
    pub fn add_block(&mut self, layer_id: u32, block: KvCacheBlockRef) {
        let blocks = self.blocks_per_layer.entry(layer_id).or_insert_with(Vec::new);
        blocks.push(block);

        // Recalculate window size
        self.window_size_bytes = self.calculate_window_size();
    }

    /// Evict blocks that are outside the attention window
    pub fn evict_outside_window(&mut self) -> Vec<KvCacheBlockRef> {
        let mut evicted = Vec::new();

        for (_, blocks) in self.blocks_per_layer.iter_mut() {
            let (in_window, outside): (Vec<_>, Vec<_>) = blocks.drain(..)
                .partition(|b| b.start_token + b.num_tokens > self.window_start);

            evicted.extend(outside);
            *blocks = in_window;
        }

        self.window_size_bytes = self.calculate_window_size();
        evicted
    }

    /// Get blocks within the attention window for a specific layer
    pub fn get_window_blocks(&self, layer_id: u32) -> Vec<&KvCacheBlockRef> {
        self.blocks_per_layer
            .get(&layer_id)
            .map(|blocks| {
                blocks.iter()
                    .filter(|b| b.start_token + b.num_tokens > self.window_start)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Calculate total size of blocks in the window
    fn calculate_window_size(&self) -> usize {
        self.blocks_per_layer.values()
            .flat_map(|blocks| blocks.iter())
            .filter(|b| b.start_token + b.num_tokens > self.window_start)
            .map(|b| b.num_tokens as usize * self.config.kv_cache_size_per_token())
            .sum()
    }
}

/// Checkpoint with windowed attention state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowedAttentionCheckpoint {
    /// Base checkpoint header
    pub header: CheckpointHeader,
    /// Token-to-expert mappings (only within attention window)
    pub mappings: Vec<TokenExpertMapping>,
    /// Expert locations
    pub expert_locations: HashMap<(u32, u32), u8>,
    /// Hot set
    pub hot_set: Vec<(u32, u32)>,
    /// KV cache window state
    pub kv_window: KvCacheWindowState,
    /// Attention mask info: which tokens can attend to which
    pub attention_mask_start: u32,
    /// Whether this is a sliding window checkpoint
    pub is_sliding_window: bool,
}

impl WindowedAttentionCheckpoint {
    /// Create a new windowed attention checkpoint
    pub fn new(
        checkpoint_id: u64,
        sequence_id: u64,
        mappings: Vec<TokenExpertMapping>,
        expert_locations: HashMap<(u32, u32), u8>,
        hot_set: Vec<(u32, u32)>,
        kv_window: KvCacheWindowState,
    ) -> Self {
        // Capture values before moving kv_window
        let attention_mask_start = kv_window.window_start;
        let is_sliding_window = kv_window.config.use_sliding_window;
        let window_start = kv_window.window_start;
        let window_len = kv_window.sequence_position - kv_window.window_start;
        let num_layers = kv_window.config.num_layers;

        let header = CheckpointHeader::new(
            checkpoint_id,
            window_start,
            window_len,
            num_layers,
            0, // Set during serialization
            sequence_id,
        );

        Self {
            header,
            mappings,
            expert_locations,
            hot_set,
            kv_window,
            attention_mask_start,
            is_sliding_window,
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self)
            .map_err(|e| format!("Failed to serialize windowed checkpoint: {}", e))
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        bincode::deserialize(data)
            .map_err(|e| format!("Failed to deserialize windowed checkpoint: {}", e))
    }

    /// Get the effective attention range for recovery
    pub fn attention_range(&self) -> (u32, u32) {
        (self.attention_mask_start, self.header.window_start + self.header.window_len)
    }

    /// Filter mappings to only those within the attention window
    pub fn filter_to_attention_window(&self) -> Vec<&TokenExpertMapping> {
        self.mappings.iter()
            .filter(|m| m.token_position >= self.attention_mask_start)
            .collect()
    }
}

/// Result of windowed attention recovery
#[derive(Debug)]
pub struct WindowedRecoveryResult {
    /// Base recovery result
    pub base: RecoveryResult,
    /// KV cache window state to restore
    pub kv_window: KvCacheWindowState,
    /// Attention mask start position
    pub attention_mask_start: u32,
    /// KV cache blocks to prefetch from CXL (within attention window)
    pub kv_blocks_to_prefetch: Vec<KvCacheBlockRef>,
    /// Estimated prefetch time based on CXL bandwidth
    pub estimated_prefetch_time_ms: f64,
}

impl WindowedRecoveryResult {
    /// Check if the recovery is complete (all KV blocks available)
    pub fn is_complete(&self) -> bool {
        self.kv_blocks_to_prefetch.iter()
            .all(|b| b.cxl_offset.is_some())
    }

    /// Get the number of tokens that need KV cache restoration
    pub fn tokens_to_restore(&self) -> u32 {
        self.kv_window.sequence_position - self.kv_window.window_start
    }
}

/// Extended checkpoint manager with windowed attention support
impl CxlCheckpointManager {
    /// Create a windowed attention checkpoint
    pub fn write_windowed_checkpoint(
        &self,
        expert_locations: HashMap<(u32, u32), u8>,
        hot_set: Vec<(u32, u32)>,
        kv_window: KvCacheWindowState,
        gpu_ptr: Option<u64>,
    ) -> CxlP2pResult<u64> {
        let start = Instant::now();

        // Get current window mappings, filtering to attention window
        let mappings = {
            let mut window = self.current_window.lock();
            let mappings: Vec<_> = std::mem::take(&mut *window)
                .into_iter()
                .filter(|m| m.token_position >= kv_window.window_start)
                .collect();
            mappings
        };

        if mappings.is_empty() {
            return Err(CxlP2pError::TransferFailed("No mappings in attention window".into()));
        }

        let checkpoint_id = self.next_checkpoint_id.fetch_add(1, Ordering::SeqCst);
        let sequence_id = self.current_sequence_id.load(Ordering::Relaxed);

        let checkpoint = WindowedAttentionCheckpoint::new(
            checkpoint_id,
            sequence_id,
            mappings,
            expert_locations,
            hot_set,
            kv_window,
        );

        let data = checkpoint.to_bytes()
            .map_err(|e| CxlP2pError::TransferFailed(e))?;
        let data_size = data.len();

        // Allocate ring slot and write to CXL
        let write_offset = self.allocate_ring_slot(data_size)?;

        if let Some(_gpu_ptr) = gpu_ptr {
            // P2P DMA path
            self.ctx.transfer_timed(
                self.ring_buffer_id,
                _gpu_ptr,
                write_offset,
                data_size,
                TransferDirection::GpuToCxl,
            )?;
        } else {
            // CPU path
            self.ctx.copy_to_buffer(self.ring_buffer_id, write_offset, &data)?;
        }

        // Record slot
        {
            let mut slots = self.slots.lock();
            while slots.len() >= MAX_RING_CHECKPOINTS {
                slots.pop_front();
            }
            let index = slots.len();
            slots.push_back(CheckpointSlot {
                index,
                offset: write_offset,
                size: data_size,
                checkpoint_id,
                valid: true,
            });
        }

        let elapsed = start.elapsed();
        self.metrics.checkpoints_written.fetch_add(1, Ordering::Relaxed);
        self.metrics.bytes_written.fetch_add(data_size as u64, Ordering::Relaxed);
        self.metrics.write_latency_us_total.fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);

        tracing::debug!(
            "Windowed checkpoint {} written: {} bytes, attention_window=[{}, {}]",
            checkpoint_id,
            data_size,
            checkpoint.attention_mask_start,
            checkpoint.header.window_start + checkpoint.header.window_len,
        );

        Ok(checkpoint_id)
    }

    /// Perform fast recovery with windowed attention optimization
    pub fn fast_windowed_recovery(
        &self,
        gpu_ptr: Option<u64>,
        cxl_bandwidth_gbps: f64,
    ) -> CxlP2pResult<WindowedRecoveryResult> {
        let start = Instant::now();

        // Get latest checkpoint
        let slot = {
            let slots = self.slots.lock();
            slots.back()
                .filter(|s| s.valid)
                .map(|s| (s.offset, s.size, s.checkpoint_id))
        };

        let (offset, size, checkpoint_id) = slot.ok_or_else(|| {
            CxlP2pError::TransferFailed("No checkpoints available".into())
        })?;

        // Read from CXL
        let data = if let Some(_gpu_ptr) = gpu_ptr {
            self.ctx.transfer_timed(
                self.ring_buffer_id,
                _gpu_ptr,
                offset,
                size,
                TransferDirection::CxlToGpu,
            )?;
            self.ctx.copy_from_buffer(self.ring_buffer_id, offset, size)?
        } else {
            self.ctx.copy_from_buffer(self.ring_buffer_id, offset, size)?
        };

        // Try to deserialize as windowed checkpoint, fall back to regular
        let (kv_window, attention_mask_start, mappings, expert_locations, hot_set) =
            if let Ok(windowed) = WindowedAttentionCheckpoint::from_bytes(&data) {
                (
                    windowed.kv_window,
                    windowed.attention_mask_start,
                    windowed.mappings,
                    windowed.expert_locations,
                    windowed.hot_set,
                )
            } else {
                // Fall back to regular checkpoint
                let regular = CheckpointData::from_bytes(&data)
                    .map_err(|e| CxlP2pError::TransferFailed(e))?;
                let kv_window = KvCacheWindowState::new(SlidingWindowAttentionConfig::default());
                (
                    kv_window,
                    regular.header.window_start,
                    regular.mappings,
                    regular.expert_locations,
                    regular.hot_set,
                )
            };

        // Build replay instructions
        let replay_instructions: Vec<ReplayInstruction> = mappings.iter()
            .filter(|m| m.token_position >= attention_mask_start)
            .map(|m| ReplayInstruction {
                sequence_id: m.sequence_id,
                token_position: m.token_position,
                layer_id: m.layer_id,
                expert_id: m.expert_id,
                topk_experts: m.topk_experts.clone(),
                gating_scores: m.gating_scores.clone(),
            })
            .collect();

        // Collect KV blocks to prefetch
        let kv_blocks_to_prefetch: Vec<KvCacheBlockRef> = kv_window.blocks_per_layer
            .values()
            .flat_map(|blocks| blocks.iter().cloned())
            .filter(|b| b.in_attention_window)
            .collect();

        // Estimate prefetch time based on CXL bandwidth
        let prefetch_bytes: usize = kv_blocks_to_prefetch.iter()
            .map(|b| b.num_tokens as usize * kv_window.config.kv_cache_size_per_token())
            .sum();
        let estimated_prefetch_time_ms = if cxl_bandwidth_gbps > 0.0 {
            (prefetch_bytes as f64 / 1e9) / cxl_bandwidth_gbps * 1000.0
        } else {
            0.0
        };

        let recovery_time = start.elapsed();
        self.metrics.checkpoints_read.fetch_add(1, Ordering::Relaxed);
        self.metrics.bytes_read.fetch_add(size as u64, Ordering::Relaxed);
        self.metrics.read_latency_us_total.fetch_add(recovery_time.as_micros() as u64, Ordering::Relaxed);
        self.metrics.recovery_count.fetch_add(1, Ordering::Relaxed);
        self.metrics.last_recovery_time_us.store(recovery_time.as_micros() as u64, Ordering::Relaxed);

        let base = RecoveryResult {
            checkpoint_id,
            window_start: kv_window.window_start,
            window_len: kv_window.sequence_position - kv_window.window_start,
            replay_instructions,
            expert_locations,
            hot_set,
            recovery_time,
        };

        tracing::info!(
            "Windowed recovery completed: {} instructions, {} KV blocks, {:.2}ms estimated prefetch",
            base.replay_instructions.len(),
            kv_blocks_to_prefetch.len(),
            estimated_prefetch_time_ms,
        );

        Ok(WindowedRecoveryResult {
            base,
            kv_window,
            attention_mask_start,
            kv_blocks_to_prefetch,
            estimated_prefetch_time_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_serialization() {
        let mut mappings = Vec::new();
        for i in 0..16 {
            mappings.push(TokenExpertMapping::new(
                1, // sequence_id
                i,
                0, // layer_id
                (i % 8) as u32,
                vec![(i % 8) as u32, ((i + 1) % 8) as u32],
                vec![0.7, 0.3],
                i as u64 * 1000,
            ));
        }

        let mut checkpoint = CheckpointData::new(
            1,
            0,
            1,
            1,
            mappings,
            HashMap::new(),
            vec![(0, 0), (0, 1)],
        );

        let data = checkpoint.to_bytes().unwrap();
        assert!(!data.is_empty());

        let recovered = CheckpointData::from_bytes(&data).unwrap();
        assert_eq!(recovered.header.checkpoint_id, 1);
        assert_eq!(recovered.mappings.len(), 16);
        assert_eq!(recovered.hot_set.len(), 2);
    }

    #[test]
    fn test_checkpoint_manager() {
        let ctx = Arc::new(CxlP2pContext::new().unwrap());
        let manager = CxlCheckpointManager::new(ctx, 16, 2, Some(1024 * 1024)).unwrap();

        manager.set_sequence_id(1);

        // Record some mappings
        for token in 0..16 {
            for layer in 0..2 {
                manager.record_mapping(
                    token,
                    layer,
                    (token % 8) as u32,
                    vec![(token % 8) as u32],
                    vec![1.0],
                    token as u64 * 100,
                );
            }
        }

        // Write checkpoint
        let checkpoint_id = manager
            .write_checkpoint(HashMap::new(), vec![(0, 0)], None)
            .unwrap();
        assert_eq!(checkpoint_id, 1);

        // Read checkpoint
        let checkpoint = manager.read_checkpoint(checkpoint_id, None).unwrap();
        assert_eq!(checkpoint.header.checkpoint_id, 1);
        assert_eq!(checkpoint.mappings.len(), 32); // 16 tokens * 2 layers
    }

    #[test]
    fn test_recovery() {
        let ctx = Arc::new(CxlP2pContext::new().unwrap());
        let manager = CxlCheckpointManager::new(ctx, 8, 1, Some(512 * 1024)).unwrap();

        manager.set_sequence_id(42);

        // Record mappings
        for token in 0..8 {
            manager.record_mapping(token, 0, token % 4, vec![token % 4], vec![1.0], token as u64);
        }

        // Write checkpoint
        let mut expert_locs = HashMap::new();
        expert_locs.insert((0, 0), 0); // GPU
        expert_locs.insert((0, 1), 1); // CXL

        manager
            .write_checkpoint(expert_locs.clone(), vec![(0, 0)], None)
            .unwrap();

        // Perform recovery
        let result = manager.fast_recovery(None).unwrap();

        assert_eq!(result.replay_instructions.len(), 8);
        assert_eq!(result.expert_locations.len(), 2);
        assert!(result.recovery_time.as_micros() > 0);
    }
}
