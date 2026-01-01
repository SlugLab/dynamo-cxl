# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CXL Checkpoint Mixin for SGLang handlers.

Provides sub-second checkpoint/recovery (~1s C/R) for MoE models by:
1. Recording expert routing decisions during inference
2. Storing checkpoints in CXL memory with P2P DMA
3. Fast replay-based recovery without re-prefill

Usage:
    class CxlDecodeHandler(CxlCheckpointMixin, DecodeWorkerHandler):
        pass
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

try:
    from dynamo._core import (
        CxlCheckpointManager,
        CxlExpertManager,
        CxlExpertManagerConfig,
    )
    CXL_AVAILABLE = True
except ImportError:
    CXL_AVAILABLE = False
    logging.warning("CXL bindings not available, using simulation mode")


@dataclass
class CxlCheckpointConfig:
    """Configuration for CXL checkpoint integration."""

    # Enable CXL checkpoint functionality
    enabled: bool = False

    # Number of experts per MoE layer
    num_experts: int = 128

    # Number of MoE layers in the model
    num_moe_layers: int = 32

    # Expert weight size in bytes (default 256MB)
    expert_weight_size: int = 256 * 1024 * 1024

    # Maximum experts to keep in GPU HBM (hot tier)
    max_gpu_experts: int = 32

    # Checkpoint window size in tokens
    window_size: int = 16

    # CXL bandwidth in Gbps (for transfer time estimation)
    cxl_bandwidth_gbps: float = 128.0

    # Checkpoint buffer size in MB
    checkpoint_buffer_mb: int = 256

    # Enable P2P DMA (requires hardware support)
    enable_p2p_dma: bool = True

    # QoS priority levels for expert prefetch
    qos_priority_levels: int = 4

    # GPU memory capacity for experts (default 80GB)
    gpu_expert_capacity: int = 80 * 1024 * 1024 * 1024

    # CXL memory capacity for experts (default 512GB)
    cxl_expert_capacity: int = 512 * 1024 * 1024 * 1024

    # Auto-checkpoint after N tokens (0 = disabled)
    auto_checkpoint_interval: int = 0

    # Commit checkpoint on sequence end
    checkpoint_on_eos: bool = True

    @classmethod
    def from_args(cls, args) -> "CxlCheckpointConfig":
        """Create config from argparse namespace."""
        return cls(
            enabled=getattr(args, "enable_cxl_checkpoint", False),
            num_experts=getattr(args, "cxl_num_experts", 128),
            num_moe_layers=getattr(args, "cxl_num_moe_layers", 32),
            max_gpu_experts=getattr(args, "cxl_max_gpu_experts", 32),
            window_size=getattr(args, "cxl_window_size", 16),
            cxl_bandwidth_gbps=getattr(args, "cxl_bandwidth_gbps", 128.0),
            checkpoint_buffer_mb=getattr(args, "cxl_checkpoint_buffer_mb", 256),
            enable_p2p_dma=getattr(args, "cxl_enable_p2p_dma", True),
            auto_checkpoint_interval=getattr(args, "cxl_auto_checkpoint_interval", 0),
            checkpoint_on_eos=getattr(args, "cxl_checkpoint_on_eos", True),
        )


@dataclass
class CxlCheckpointState:
    """Runtime state for CXL checkpointing."""

    # Current sequence ID being processed
    current_sequence_id: int = 0

    # Token counter for current sequence
    token_counter: int = 0

    # Tokens since last checkpoint
    tokens_since_checkpoint: int = 0

    # Number of checkpoints created
    checkpoint_count: int = 0

    # Recovery count
    recovery_count: int = 0

    # Total recovery time in ms
    total_recovery_time_ms: float = 0.0

    # Expert locations: (layer_id, expert_id) -> location (0=GPU, 1=CXL)
    expert_locations: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # Hot set of experts (in GPU)
    hot_set: Set[Tuple[int, int]] = field(default_factory=set)


class CxlCheckpointMixin:
    """
    Mixin that adds CXL checkpoint functionality to SGLang handlers.

    This mixin intercepts inference to record expert routing decisions
    and provides checkpoint/recovery functionality for fault tolerance.

    Architecture:
        GPU HBM (hot tier): Active experts, hot KV cache
        CXL Memory (cold tier): Parked experts, checkpoint storage
        P2P DMA: GPU <-> CXL bypass CPU for fast transfers

    Recovery Flow:
        1. Read checkpoint from CXL memory
        2. Replay routing decisions (expert selection)
        3. Resume inference from last checkpoint window
        No re-prefill needed - KV cache in CXL, just restore routing state
    """

    def __init__(self, *args, cxl_config: Optional[CxlCheckpointConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.cxl_config = cxl_config or CxlCheckpointConfig()
        self.cxl_state = CxlCheckpointState()

        self._checkpoint_manager: Optional["CxlCheckpointManager"] = None
        self._expert_manager: Optional["CxlExpertManager"] = None

        if self.cxl_config.enabled:
            self._init_cxl_managers()

    def _init_cxl_managers(self):
        """Initialize CXL checkpoint and expert managers."""
        if not CXL_AVAILABLE:
            logging.warning("CXL bindings not available, using simulation mode")
            return

        try:
            # Initialize checkpoint manager
            self._checkpoint_manager = CxlCheckpointManager(
                window_size=self.cxl_config.window_size,
                num_layers=self.cxl_config.num_moe_layers,
                buffer_size_mb=self.cxl_config.checkpoint_buffer_mb,
            )

            # Initialize expert manager
            expert_config = CxlExpertManagerConfig(
                num_experts=self.cxl_config.num_experts,
                num_moe_layers=self.cxl_config.num_moe_layers,
                expert_weight_size=self.cxl_config.expert_weight_size,
                gpu_expert_capacity=self.cxl_config.gpu_expert_capacity,
                cxl_expert_capacity=self.cxl_config.cxl_expert_capacity,
                window_size=self.cxl_config.window_size,
                cxl_bandwidth_gbps=self.cxl_config.cxl_bandwidth_gbps,
                max_gpu_experts=self.cxl_config.max_gpu_experts,
                enable_p2p_dma=self.cxl_config.enable_p2p_dma,
                qos_priority_levels=self.cxl_config.qos_priority_levels,
            )
            self._expert_manager = CxlExpertManager(expert_config)

            # Register experts with initial locations
            self._register_experts()

            logging.info(
                f"CXL checkpoint initialized: "
                f"{self.cxl_config.num_experts} experts x {self.cxl_config.num_moe_layers} layers, "
                f"max {self.cxl_config.max_gpu_experts} in GPU, "
                f"window size {self.cxl_config.window_size}"
            )
        except Exception as e:
            logging.error(f"Failed to initialize CXL managers: {e}")
            self._checkpoint_manager = None
            self._expert_manager = None

    def _register_experts(self):
        """Register all experts with initial GPU/CXL placement."""
        if not self._expert_manager:
            return

        for layer_id in range(self.cxl_config.num_moe_layers):
            for expert_id in range(self.cxl_config.num_experts):
                # First max_gpu_experts go to GPU, rest to CXL
                location = "gpu" if expert_id < self.cxl_config.max_gpu_experts else "cxl"

                self._expert_manager.register_expert(
                    expert_id=expert_id,
                    layer_id=layer_id,
                    weight_size_bytes=self.cxl_config.expert_weight_size,
                    initial_location=location,
                )

                key = (layer_id, expert_id)
                if location == "gpu":
                    self.cxl_state.hot_set.add(key)
                    self.cxl_state.expert_locations[key] = 0
                else:
                    self.cxl_state.expert_locations[key] = 1

    def set_sequence_id(self, sequence_id: int):
        """Set the current sequence ID for checkpointing."""
        self.cxl_state.current_sequence_id = sequence_id
        self.cxl_state.token_counter = 0
        self.cxl_state.tokens_since_checkpoint = 0

        if self._checkpoint_manager:
            self._checkpoint_manager.set_sequence_id(sequence_id)

    def record_routing_decision(
        self,
        token_position: int,
        layer_id: int,
        expert_id: int,
        topk_experts: Optional[List[int]] = None,
        gating_scores: Optional[List[float]] = None,
        kv_block_hash: int = 0,
    ):
        """
        Record an expert routing decision for checkpointing.

        This should be called after each token is processed through an MoE layer.

        Args:
            token_position: Position of the token in the sequence
            layer_id: MoE layer ID
            expert_id: Selected expert ID
            topk_experts: List of top-k expert IDs (optional)
            gating_scores: Gating scores for top-k (optional)
            kv_block_hash: Hash of KV block for reference (optional)
        """
        if not self.cxl_config.enabled:
            return

        topk = topk_experts or [expert_id]
        scores = gating_scores or [1.0]

        # Record in checkpoint manager
        if self._checkpoint_manager:
            self._checkpoint_manager.record_mapping(
                token_position=token_position,
                layer_id=layer_id,
                expert_id=expert_id,
                topk_experts=topk,
                gating_scores=scores,
                kv_block_hash=kv_block_hash,
            )

        # Record in expert manager for KV delta
        if self._expert_manager:
            tokens_hash = hash((self.cxl_state.current_sequence_id, token_position))
            # Ensure hash values are positive (convert to unsigned)
            tokens_hash_unsigned = tokens_hash & 0xFFFFFFFFFFFFFFFF
            kv_block_hash_unsigned = kv_block_hash & 0xFFFFFFFFFFFFFFFF
            self._expert_manager.record_kv_delta(
                token_offset=token_position,
                layer_id=layer_id,
                expert_id=expert_id,
                topk_experts=topk,
                gating_scores=scores,
                kv_block_hash=kv_block_hash_unsigned,
                tokens_hash=tokens_hash_unsigned,
            )

        self.cxl_state.token_counter += 1
        self.cxl_state.tokens_since_checkpoint += 1

        # Auto-checkpoint if configured
        if (self.cxl_config.auto_checkpoint_interval > 0 and
            self.cxl_state.tokens_since_checkpoint >= self.cxl_config.auto_checkpoint_interval):
            self.force_checkpoint()

    def force_checkpoint(self) -> Optional[int]:
        """
        Force a checkpoint to be written.

        Returns:
            Checkpoint ID if checkpoint was written, None otherwise.
        """
        if not self.cxl_config.enabled:
            return None

        checkpoint_id = None

        # Get expert locations as dict for checkpoint
        expert_locs = {k: v for k, v in self.cxl_state.expert_locations.items()}
        hot_set_list = list(self.cxl_state.hot_set)

        if self._checkpoint_manager:
            try:
                checkpoint_id = self._checkpoint_manager.force_commit(
                    expert_locations=expert_locs,
                    hot_set=hot_set_list,
                )
                if checkpoint_id is not None:
                    self.cxl_state.checkpoint_count += 1
                    self.cxl_state.tokens_since_checkpoint = 0
                    logging.debug(f"Checkpoint {checkpoint_id} written")
            except Exception as e:
                logging.warning(f"Failed to write checkpoint: {e}")

        if self._expert_manager:
            try:
                self._expert_manager.force_checkpoint()
            except Exception as e:
                logging.warning(f"Failed to force expert checkpoint: {e}")

        return checkpoint_id

    async def fast_recovery(self, gpu_ptr: Optional[int] = None) -> Dict:
        """
        Perform fast recovery from the latest checkpoint.

        This reads the checkpoint from CXL memory and returns the routing
        decisions that need to be replayed. No re-prefill needed.

        Args:
            gpu_ptr: Optional GPU pointer for P2P DMA (None for CPU path)

        Returns:
            Dict with recovery information including:
            - checkpoint_id: ID of checkpoint recovered from
            - window_start: Start token of recovery window
            - window_len: Number of tokens in window
            - replay_instructions: List of routing decisions to replay
            - recovery_time_us: Recovery time in microseconds
        """
        if not self.cxl_config.enabled or not self._checkpoint_manager:
            return {"error": "CXL checkpoint not enabled"}

        start_time = time.time()

        try:
            result = self._checkpoint_manager.fast_recovery(gpu_ptr=gpu_ptr)

            recovery_time_ms = (time.time() - start_time) * 1000
            self.cxl_state.recovery_count += 1
            self.cxl_state.total_recovery_time_ms += recovery_time_ms

            # Restore expert locations from checkpoint
            if "expert_locations" in result:
                for key_str, loc in result["expert_locations"].items():
                    parts = key_str.split("_")
                    if len(parts) == 2:
                        key = (int(parts[0]), int(parts[1]))
                        self.cxl_state.expert_locations[key] = loc

            if "hot_set" in result:
                self.cxl_state.hot_set = set(tuple(x) for x in result["hot_set"])

            logging.info(
                f"Fast recovery completed in {recovery_time_ms:.2f}ms: "
                f"checkpoint {result.get('checkpoint_id')}, "
                f"{len(result.get('replay_instructions', []))} routing decisions"
            )

            return result

        except Exception as e:
            logging.error(f"Fast recovery failed: {e}")
            return {"error": str(e)}

    async def request_expert(
        self,
        expert_id: int,
        layer_id: int,
        priority: int = 0,
    ) -> bool:
        """
        Request an expert for inference.

        If the expert is not in GPU (cache miss), this will trigger
        a prefetch from CXL memory with P2P DMA.

        Args:
            expert_id: Expert ID to request
            layer_id: Layer ID
            priority: QoS priority (higher = more urgent)

        Returns:
            True if expert is available for inference
        """
        if not self._expert_manager:
            return True  # No manager, assume all experts available

        try:
            return await self._expert_manager.request_expert(
                expert_id=expert_id,
                layer_id=layer_id,
                priority=priority,
            )
        except Exception as e:
            logging.warning(f"Expert request failed: {e}")
            return False

    def get_cxl_metrics(self) -> Dict:
        """Get CXL checkpoint and expert manager metrics."""
        metrics = {
            "enabled": self.cxl_config.enabled,
            "checkpoint_count": self.cxl_state.checkpoint_count,
            "recovery_count": self.cxl_state.recovery_count,
            "avg_recovery_time_ms": (
                self.cxl_state.total_recovery_time_ms / max(1, self.cxl_state.recovery_count)
            ),
            "tokens_processed": self.cxl_state.token_counter,
            "gpu_experts": len(self.cxl_state.hot_set),
        }

        if self._checkpoint_manager:
            try:
                ckpt_metrics = self._checkpoint_manager.get_metrics()
                metrics.update({
                    "checkpoints_written": ckpt_metrics.get("checkpoints_written", 0),
                    "checkpoints_read": ckpt_metrics.get("checkpoints_read", 0),
                    "checkpoint_bytes_written": ckpt_metrics.get("bytes_written", 0),
                    "avg_checkpoint_write_latency_us": ckpt_metrics.get("avg_write_latency_us", 0),
                    "avg_checkpoint_read_latency_us": ckpt_metrics.get("avg_read_latency_us", 0),
                })
            except Exception:
                pass

        if self._expert_manager:
            try:
                expert_metrics = self._expert_manager.get_metrics()
                metrics.update({
                    "cache_hits": expert_metrics.get("cache_hits", 0),
                    "cache_misses": expert_metrics.get("cache_misses", 0),
                    "cache_hit_rate": expert_metrics.get("cache_hit_rate", 0),
                    "total_prefetches": expert_metrics.get("total_prefetches", 0),
                    "total_evictions": expert_metrics.get("total_evictions", 0),
                })
            except Exception:
                pass

        return metrics

    def cleanup_cxl(self):
        """Cleanup CXL resources."""
        if self._expert_manager:
            try:
                self._expert_manager.shutdown()
            except Exception as e:
                logging.warning(f"Error shutting down expert manager: {e}")

        self._checkpoint_manager = None
        self._expert_manager = None
        logging.info("CXL resources cleaned up")
