// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use anyhow::Result;
use clap::{ArgAction, Parser, ValueEnum};
use dynamo_llm::kv_router::{indexer::KvIndexerMetrics, wal::{MoeWalEvent, MoeWalReplayer}};
use dynamo_llm::kv_router::{indexer::{KvIndexer, KvIndexerInterface, RouterEvent}, KV_EVENT_SUBJECT};
use dynamo_runtime::transports::nats::{NatsQueue, Slug};
use dynamo_runtime::traits::events::EventPublisher;
use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::fs::File;
use tokio_util::sync::CancellationToken;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Mode { Local, Nats, Stdout }

#[derive(Parser, Debug)]
#[command(version, about = "Replay a MoE WAL (record->replay)")]
struct Args {
    /// Path to WAL JSONL file
    #[arg(long)]
    wal: PathBuf,

    /// Mode: local indexer (dry-run), publish to NATS JetStream, or write RouterEvents to stdout
    #[arg(long, value_enum, default_value = "local")]
    mode: Mode,

    /// Namespace name (required for --mode nats)
    #[arg(long)]
    namespace: Option<String>,

    /// Worker id to stamp on RouterEvents
    #[arg(long, default_value_t = 0)]
    worker_id: i64,

    /// Replay respecting recorded inter-record delays
    #[arg(long, action = ArgAction::SetTrue, default_value_t = false)]
    timed: bool,

    /// Max events to process
    #[arg(long)]
    max_events: Option<usize>,

    /// Max time to process (seconds)
    #[arg(long)]
    max_time_secs: Option<f64>,

    /// NATS server URL (for mode=nats)
    #[arg(long, default_value = "nats://localhost:4222")]
    nats_server: String,
}

#[derive(Deserialize)]
struct WalLine { timestamp: u64, event: MoeWalEvent }

#[tokio::main]
async fn main() -> Result<()> {
    dynamo_runtime::logging::init();
    let args = Args::parse();

    match args.mode {
        Mode::Local => replay_local(args).await,
        Mode::Nats => replay_nats(args).await,
        Mode::Stdout => dump_stdout(args).await,
    }
}

async fn dump_stdout(args: Args) -> Result<()> {
    let file = File::open(&args.wal).await?;
    let reader = BufReader::with_capacity(32768, file);
    let mut lines = reader.lines();
    let mut out = tokio::io::stdout();
    let mut count = 0usize;
    while let Some(line) = lines.next_line().await? {
        if let Ok(line) = serde_json::from_str::<WalLine>(&line) {
            if let MoeWalEvent::Record(r) = line.event {
                for e in MoeWalReplayer::record_to_events(args.worker_id, &r) {
                    let json = serde_json::to_string(&e)?;
                    out.write_all(json.as_bytes()).await?;
                    out.write_all(b"\n").await?;
                    count += 1;
                    if let Some(max) = args.max_events { if count >= max { break; } }
                }
            }
        }
        if let Some(max) = args.max_events { if count >= max { break; } }
    }
    tracing::info!("wrote {} RouterEvents to stdout", count);
    Ok(())
}

async fn replay_local(args: Args) -> Result<()> {
    let cancel = CancellationToken::new();
    let metrics = std::sync::Arc::new(KvIndexerMetrics::new_unregistered());
    let mut indexer = KvIndexer::new(cancel.clone(), 32, metrics);
    let applied = MoeWalReplayer::replay_to_indexer(
        &args.wal,
        &mut indexer,
        args.worker_id,
        args.timed,
        args.max_events,
        args.max_time_secs,
    ).await?;
    tracing::info!("applied {} events to local indexer", applied);
    Ok(())
}

async fn replay_nats(args: Args) -> Result<()> {
    let ns = args.namespace.clone().ok_or_else(|| anyhow::anyhow!("--namespace required for mode nats"))?;
    // Build the exact stream name expected by the router background task
    let ns_subject = format!("namespace.{}", ns);
    let stream = Slug::slugify(&format!("{}.{}", ns_subject, KV_EVENT_SUBJECT)).to_string().replace('_', "-");
    let mut queue = NatsQueue::new_without_consumer(stream.clone(), args.nats_server.clone(), std::time::Duration::from_secs(60));
    queue.connect().await?;

    let file = File::open(&args.wal).await?;
    let reader = BufReader::with_capacity(32768, file);
    let mut lines = reader.lines();
    let mut count = 0usize;
    while let Some(line) = lines.next_line().await? {
        if let Ok(line) = serde_json::from_str::<WalLine>(&line) {
            if let MoeWalEvent::Record(r) = line.event {
                for e in MoeWalReplayer::record_to_events(args.worker_id, &r) {
                    queue.publish("queue", &e).await?;
                    count += 1;
                    if let Some(max) = args.max_events { if count >= max { break; } }
                }
            }
        }
        if let Some(max) = args.max_events { if count >= max { break; } }
    }
    tracing::info!("published {} RouterEvents to NATS stream {}", count, stream);
    Ok(())
}
