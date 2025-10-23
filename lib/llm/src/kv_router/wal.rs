// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MoE Write-Ahead Logging (WAL) for fast, fault-tolerant recovery.
//!
//! This module provides a minimal, zero-copy-friendly log of MoE routing decisions
//! and KV cache deltas at a configurable token window size. The log is designed to
//! enable record→replay recovery without token-wise checkpointing.
//!
//! Design goals
//! - Minimal state: record gating decisions (expert assignments) and KV deltas
//!   by reference using existing `KvCacheStoredBlockData`.
//! - Window-based: amortize logging cost over a small token window (e.g. 16).
//! - Replay-friendly: reconstruct token→expert mapping and re-apply KV deltas
//!   via the existing router/indexer event path.
//!
//! Typical usage (writer):
//! - Instantiate a `MoeWalWriter` with a file path and window size.
//! - At the end of each inference window, call `log_record(...)` with expert
//!   assignments and any newly materialized KV blocks (as `KvCacheStoredBlockData`).
//!
//! Typical usage (replayer):
//! - Construct a `MoeWalReplayer` from a WAL file and call
//!   `replay_to_indexer(...)` to convert WAL records to RouterEvents and apply
//!   them to a live `KvIndexerInterface` instance.

use super::indexer::{KvIndexerInterface, RouterEvent};
use super::protocols::{
    CxlPoolId, ExternalSequenceBlockHash, KvCacheBlockMoEMetadata, KvCacheEvent, KvCacheEventData,
    KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash,
};
use crate::recorder::Recorder;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// WAL header describing session and static parameters that affect logging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeWalHeader {
    /// Stable identifier for the inference session.
    pub session_id: Uuid,
    /// Window size used for logging (tokens per record).
    pub window_size: u32,
    /// Optional model identifier or tag for bookkeeping.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_tag: Option<String>,
}

/// Compact per-window record of MoE routing and KV deltas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeWalRecord {
    /// Inference sequence UUID.
    pub sequence_id: Uuid,
    /// Layer where routing occurred.
    pub layer_id: u32,
    /// Start token offset of the window within the sequence.
    pub window_start: u32,
    /// Number of tokens included in this record (<= window_size).
    pub window_len: u32,
    /// Selected expert per token position in the window.
    pub expert_ids: Vec<u32>,
    /// Optional: per-token top-k expert ids. If present, scores are parallel to ids.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topk_expert_ids: Option<Vec<Vec<u32>>>,
    /// Optional: per-token top-k gating scores (implementation-defined scale).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topk_gating_scores: Option<Vec<Vec<f32>>>,
    /// Token indices (relative to the full sequence) covered by this window.
    /// If omitted, treated as `[window_start, window_start + window_len)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_indices: Option<Vec<u32>>,
    /// Optional hint for the active CXL pool during prefill→decode transitions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cxl_pool_hint: Option<CxlPoolId>,
    /// Referenced KV deltas materialized during this window.
    /// Blocks are referenced by hash and token hash; payloads are not duplicated.
    #[serde(default)]
    pub kv_deltas: Vec<KvCacheStoredBlockData>,
}

/// Checkpoint marker for faster replay seeking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeWalCheckpoint {
    /// Byte offset hint into the WAL (filled by the writer for quick seeks).
    pub wal_offset: Option<u64>,
    /// Map of sequence_id to last fully applied window_start for resumption.
    pub last_committed_windows: HashMap<Uuid, u32>,
}

/// Top-level WAL event type persisted as JSONL via `Recorder`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum MoeWalEvent {
    Header(MoeWalHeader),
    Record(MoeWalRecord),
    Checkpoint(MoeWalCheckpoint),
}

/// Write-ahead log writer. Thin wrapper over the generic `Recorder`.
pub struct MoeWalWriter {
    header: MoeWalHeader,
    recorder: Recorder<MoeWalEvent>,
    tx: mpsc::Sender<MoeWalEvent>,
}

impl MoeWalWriter {
    /// Create a new writer and emit a Header as the first line.
    pub async fn new<P: AsRef<Path>>(
        session_id: Uuid,
        window_size: u32,
        output_path: P,
        cancel: CancellationToken,
        model_tag: Option<String>,
    ) -> Result<Self> {
        let header = MoeWalHeader {
            session_id,
            window_size,
            model_tag,
        };

        let recorder = Recorder::<MoeWalEvent>::new(cancel.clone(), output_path, None, None, None)
            .await
            .context("create WAL recorder")?;
        let tx = recorder.event_sender();

        // Emit header once.
        tx.send(MoeWalEvent::Header(header.clone()))
            .await
            .context("emit WAL header")?;

        Ok(Self { header, recorder, tx })
    }

    /// Log a per-window record.
    pub async fn log_record(&self, record: MoeWalRecord) -> Result<()> {
        // Basic shape validation to catch logging mismatches early.
        if record.window_len as usize != record.expert_ids.len() {
            anyhow::bail!("expert_ids length must equal window_len");
        }
        if let Some(r) = &record.routing_indices {
            if r.len() != record.window_len as usize {
                anyhow::bail!("routing_indices length must equal window_len");
            }
        }
        self.tx
            .send(MoeWalEvent::Record(record))
            .await
            .context("emit WAL record")
    }

    /// Log a sequence of expert assignments using writer's window size.
    /// Splits into N records of size `self.header.window_size` (last may be partial).
    pub async fn log_windowed(
        &self,
        sequence_id: Uuid,
        layer_id: u32,
        window_start: u32,
        expert_ids: &[u32],
        routing_indices: Option<&[u32]>,
        cxl_pool_hint: Option<CxlPoolId>,
        kv_deltas: Vec<KvCacheStoredBlockData>,
    ) -> Result<()> {
        let ws = self.header.window_size as usize;
        if let Some(ri) = routing_indices {
            if ri.len() != expert_ids.len() {
                anyhow::bail!("routing_indices length must equal expert_ids length");
            }
        }
        let mut offset = 0usize;
        while offset < expert_ids.len() {
            let remain = expert_ids.len() - offset;
            let take = remain.min(ws);
            let slice = &expert_ids[offset..offset + take];
            let ri_slice = routing_indices.map(|ri| &ri[offset..offset + take]);
            let record = MoeWalRecord {
                sequence_id,
                layer_id,
                window_start: window_start + offset as u32,
                window_len: take as u32,
                expert_ids: slice.to_vec(),
                topk_expert_ids: None,
                topk_gating_scores: None,
                routing_indices: ri_slice.map(|v| v.to_vec()),
                cxl_pool_hint,
                kv_deltas: if offset == 0 { kv_deltas.clone() } else { Vec::new() },
            };
            self.log_record(record).await?;
            offset += take;
        }
        Ok(())
    }

    /// Periodically emit a checkpoint to speed up replay seeking.
    pub async fn checkpoint(&self, last_committed: HashMap<Uuid, u32>) -> Result<()> {
        let evt = MoeWalEvent::Checkpoint(MoeWalCheckpoint {
            wal_offset: None,
            last_committed_windows: last_committed,
        });
        self.tx
            .send(evt)
            .await
            .context("emit WAL checkpoint")
    }

    /// Accessor for header.
    pub fn header(&self) -> &MoeWalHeader { &self.header }

    /// Stop the writer and flush pending records.
    pub fn shutdown(&self) { self.recorder.shutdown(); }
}

/// Streaming WAL replayer.
pub struct MoeWalReplayer;

impl MoeWalReplayer {
    /// Convert a WAL record into one or more RouterEvents to rehydrate the trie.
    pub fn record_to_events(worker_id: i64, record: &MoeWalRecord) -> Vec<RouterEvent> {
        // Heuristic: annotate delta blocks with the dominant expert in the window.
        let dominant_expert = Self::mode_expert(&record.expert_ids).unwrap_or(0);
        let moe_meta = KvCacheBlockMoEMetadata::new(record.layer_id, dominant_expert, None);

        let blocks: Vec<KvCacheStoredBlockData> = record
            .kv_deltas
            .iter()
            .cloned()
            .map(|mut b| {
                // Attach MoE metadata if absent so replay remains MoE-aware.
                if b.moe_metadata.is_none() {
                    let md = moe_meta.clone();
                    KvCacheStoredBlockData {
                        moe_metadata: Some(md),
                        ..b
                    }
                } else {
                    b
                }
            })
            .collect();

        if blocks.is_empty() {
            return Vec::new();
        }

        // Use parent=None: WAL deltas are independent for replay into the trie.
        let data = KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: None,
            blocks,
        });

        let event = KvCacheEvent {
            event_id: 0, // caller can stamp a monotonic id if desired
            data,
        };

        vec![RouterEvent::new(worker_id, event)]
    }

    fn mode_expert(expert_ids: &[u32]) -> Option<u32> {
        let mut counts: HashMap<u32, u32> = HashMap::new();
        for &e in expert_ids {
            *counts.entry(e).or_default() += 1;
        }
        counts.into_iter().max_by_key(|(_, c)| *c).map(|(e, _)| e)
    }

    /// Replay a WAL file into the provided indexer by converting records into `RouterEvent`s.
    ///
    /// If `timed` is true, respects inter-record timing encoded by the recorder; otherwise
    /// replays as fast as possible.
    pub async fn replay_to_indexer<P: AsRef<Path>, I: KvIndexerInterface + Send + Sync>(
        wal_path: P,
        indexer: &mut I,
        worker_id: i64,
        timed: bool,
        max_events: Option<usize>,
        max_time_secs: Option<f64>,
    ) -> Result<usize> {
        use tokio::fs::File;
        use tokio::io::{AsyncBufReadExt, BufReader};
        use tokio::time::{Instant, Sleep};

        #[derive(Deserialize)]
        struct WalLine { timestamp: u64, event: MoeWalEvent }

        // Open file and set up timing budget if needed
        let start_time = Instant::now();
        let deadline = max_time_secs.map(|s| start_time + std::time::Duration::from_secs_f64(s));

        let file = File::open(wal_path).await?;
        let reader = BufReader::with_capacity(32768, file);
        let mut lines = reader.lines();

        let mut applied = 0usize;
        let mut prev_ts: Option<u64> = None;

        while let Some(line) = lines.next_line().await? {
            if let Some(d) = deadline { if Instant::now() >= d { break; } }
            if let Some(max) = max_events { if applied >= max { break; } }
            if line.trim().is_empty() { continue; }

            let parsed: WalLine = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => continue };

            if timed {
                if let Some(prev) = prev_ts {
                    let dt_ms = parsed.timestamp.saturating_sub(prev);
                    if dt_ms > 0 { tokio::time::sleep(std::time::Duration::from_millis(dt_ms as u64)).await; }
                }
                prev_ts = Some(parsed.timestamp);
            }

            match parsed.event {
                MoeWalEvent::Header(_) | MoeWalEvent::Checkpoint(_) => {}
                MoeWalEvent::Record(r) => {
                    for e in Self::record_to_events(worker_id, &r) {
                        indexer.apply_event(e).await;
                        applied += 1;
                    }
                }
            }
        }

        Ok(applied)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_router::indexer::{KvIndexer, KvIndexerInterface, KvIndexerMetrics};
    use tempfile::tempdir;

    #[tokio::test]
    async fn wal_round_trip_minimal() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("wal.jsonl");

        let cancel = CancellationToken::new();
        let writer = MoeWalWriter::new(Uuid::new_v4(), 16, &path, cancel.clone(), None)
            .await
            .unwrap();

        // Fake delta for a single block
        let rec = MoeWalRecord {
            sequence_id: Uuid::new_v4(),
            layer_id: 3,
            window_start: 0,
            window_len: 4,
            expert_ids: vec![7, 7, 3, 7],
            topk_expert_ids: None,
            topk_gating_scores: None,
            routing_indices: None,
            cxl_pool_hint: None,
            kv_deltas: vec![KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(42),
                tokens_hash: LocalBlockHash(1337),
                moe_metadata: None,
                cxl_metadata: None,
            }],
        };
        writer.log_record(rec).await.unwrap();
        writer.shutdown();

        // Rebuild a fresh indexer and replay
        let cancel = CancellationToken::new();
        let metrics = std::sync::Arc::new(KvIndexerMetrics::new_unregistered());
        let mut indexer = KvIndexer::new(cancel.clone(), 32, metrics);

        let applied = MoeWalReplayer::replay_to_indexer(&path, &mut indexer, 0, false, None, None)
            .await
            .unwrap();
        assert!(applied >= 1);
    }

    #[tokio::test]
    async fn wal_windowing_split_and_replay() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("wal.jsonl");

        let cancel = CancellationToken::new();
        let writer = MoeWalWriter::new(Uuid::new_v4(), 4, &path, cancel.clone(), None)
            .await
            .unwrap();

        // 10 tokens => 3 windows: 4,4,2
        let seq = Uuid::new_v4();
        let experts: Vec<u32> = vec![7, 1, 7, 7, 2, 2, 2, 2, 3, 3];
        let routing: Vec<u32> = (0..10).collect();
        let kv = vec![KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(999),
            tokens_hash: LocalBlockHash(4242),
            moe_metadata: None,
            cxl_metadata: None,
        }];

        writer
            .log_windowed(seq, 5, 0, &experts, Some(&routing), None, kv)
            .await
            .unwrap();
        writer.shutdown();

        // Count records in file and check lengths
        let content = tokio::fs::read_to_string(&path).await.unwrap();
        let mut record_lens = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() { continue; }
            #[derive(Deserialize)]
            struct Line { event: MoeWalEvent }
            if let Ok(line) = serde_json::from_str::<Line>(line) {
                if let MoeWalEvent::Record(r) = line.event { record_lens.push(r.window_len); }
            }
        }
        assert_eq!(record_lens, vec![4, 4, 2]);

        // Replay fast and verify we applied some events
        let cancel = CancellationToken::new();
        let metrics = std::sync::Arc::new(KvIndexerMetrics::new_unregistered());
        let mut indexer = KvIndexer::new(cancel.clone(), 32, metrics);
        let applied = MoeWalReplayer::replay_to_indexer(&path, &mut indexer, 0, false, None, None)
            .await
            .unwrap();
        assert!(applied >= 1);
    }
}
