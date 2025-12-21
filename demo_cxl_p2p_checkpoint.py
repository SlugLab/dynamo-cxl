#!/usr/bin/env python3
"""
CXL P2P Checkpoint/Restore Demo

Demonstrates checkpoint/restore using CXL memory with P2P DMA for MoE models.
This shows how token-to-expert mappings are recorded and replayed for fast recovery.

Key features:
1. Record token-to-expert mappings during inference
2. Store checkpoints in CXL memory via P2P DMA
3. Fast recovery by reading from CXL and replaying routing decisions
4. Sub-millisecond recovery compared to minutes with traditional approaches
"""

import time
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import the CXL checkpoint manager from Rust bindings
try:
    from dynamo._core import CxlCheckpointManager, CxlExpertManagerConfig, CxlExpertManager
    HAS_CXL_BINDINGS = True
except ImportError:
    print("Warning: CXL bindings not available, using simulation mode")
    HAS_CXL_BINDINGS = False


@dataclass
class InferenceContext:
    """Context for tracking inference state"""
    sequence_id: int
    tokens_processed: int
    current_window: int
    expert_routing_history: List[Tuple[int, int, int]]  # (token, layer, expert)


class CxlP2pCheckpointDemo:
    """
    Demonstrates CXL P2P checkpoint/restore for MoE models.

    This demo shows:
    1. Recording token-to-expert mappings during simulated inference
    2. Storing checkpoints in CXL memory
    3. Fast recovery from CXL checkpoints
    4. Comparison with traditional checkpoint approaches
    """

    def __init__(self,
                 num_experts: int = 128,
                 num_layers: int = 32,
                 window_size: int = 16,
                 buffer_size_mb: int = 256):
        """Initialize the demo with model configuration."""
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.window_size = window_size
        self.buffer_size_mb = buffer_size_mb

        # Initialize CXL Checkpoint Manager
        if HAS_CXL_BINDINGS:
            print(f"\nInitializing CXL Checkpoint Manager...")
            print(f"  Window size: {window_size} tokens")
            print(f"  Num layers: {num_layers}")
            print(f"  Buffer size: {buffer_size_mb}MB")

            self.checkpoint_mgr = CxlCheckpointManager(
                window_size=window_size,
                num_layers=num_layers,
                buffer_size_mb=buffer_size_mb
            )
            print(f"  Manager: {self.checkpoint_mgr}")
        else:
            self.checkpoint_mgr = None

        # Also initialize Expert Manager for hot/cold tracking
        if HAS_CXL_BINDINGS:
            config = CxlExpertManagerConfig(
                num_experts=num_experts,
                num_moe_layers=num_layers,
                expert_weight_size=256*1024*1024,  # 256MB per expert
                max_gpu_experts=32,
                window_size=window_size,
                cxl_bandwidth_gbps=128.0,
            )
            self.expert_mgr = CxlExpertManager(config)
        else:
            self.expert_mgr = None

        self.contexts: Dict[int, InferenceContext] = {}
        self.checkpoint_count = 0

    def register_experts(self):
        """Register all experts in the system."""
        print("\n" + "="*70)
        print("PHASE 1: Expert Registration")
        print("="*70)

        total_experts = self.num_experts * self.num_layers
        gpu_experts = 32 * self.num_layers  # First 32 experts per layer in GPU
        cxl_experts = total_experts - gpu_experts

        print(f"\nModel Configuration:")
        print(f"  Total Experts: {total_experts:,}")
        print(f"  GPU HBM (Hot): {gpu_experts:,} experts")
        print(f"  CXL Memory (Cold): {cxl_experts:,} experts")

        if self.expert_mgr:
            start = time.time()
            for layer in range(self.num_layers):
                for expert in range(self.num_experts):
                    location = 'gpu' if expert < 32 else 'cxl'
                    self.expert_mgr.register_expert(
                        expert, layer,
                        256*1024*1024,  # 256MB per expert
                        location
                    )
            elapsed = (time.time() - start) * 1000
            print(f"\nRegistered {total_experts:,} experts in {elapsed:.2f}ms")
        else:
            print("\n(Simulation mode - no actual expert registration)")

    def simulate_inference(self, sequence_id: int, num_tokens: int) -> InferenceContext:
        """Simulate MoE inference with token-to-expert mapping recording."""
        print("\n" + "="*70)
        print("PHASE 2: Simulated Inference with Checkpointing")
        print("="*70)

        if self.checkpoint_mgr:
            self.checkpoint_mgr.set_sequence_id(sequence_id)

        context = InferenceContext(
            sequence_id=sequence_id,
            tokens_processed=0,
            current_window=0,
            expert_routing_history=[]
        )
        self.contexts[sequence_id] = context

        print(f"\nProcessing {num_tokens} tokens for sequence {sequence_id}")
        print(f"Checkpoint window: every {self.window_size} tokens")
        print()

        checkpoint_times = []
        start = time.time()

        for token_idx in range(num_tokens):
            # Simulate expert routing for each MoE layer
            for layer_id in range(self.num_layers):
                expert_id = self._simulate_expert_routing(token_idx, layer_id)
                topk = [expert_id, (expert_id + 1) % self.num_experts]
                gating_scores = [0.7, 0.3]

                context.expert_routing_history.append((token_idx, layer_id, expert_id))

                # Record mapping in checkpoint manager
                if self.checkpoint_mgr:
                    kv_block_hash = (token_idx * 12345 + layer_id * 67890) % (2**63)
                    self.checkpoint_mgr.record_mapping(
                        token_position=token_idx,
                        layer_id=layer_id,
                        expert_id=expert_id,
                        topk_experts=topk,
                        gating_scores=gating_scores,
                        kv_block_hash=kv_block_hash
                    )

            context.tokens_processed += 1

            # Write checkpoint at window boundary
            if (token_idx + 1) % self.window_size == 0:
                ckpt_start = time.time()

                if self.checkpoint_mgr:
                    # Get expert locations (0=GPU, 1=CXL)
                    expert_locs = {}
                    for layer in range(self.num_layers):
                        for exp in range(self.num_experts):
                            expert_locs[(layer, exp)] = 0 if exp < 32 else 1

                    # Hot set (first 32 experts per layer)
                    hot_set = [(layer, exp) for layer in range(self.num_layers)
                               for exp in range(32)]

                    checkpoint_id = self.checkpoint_mgr.write_checkpoint(
                        expert_locations=expert_locs,
                        hot_set=hot_set,
                        gpu_ptr=None  # Use CPU path for demo
                    )

                ckpt_time = (time.time() - ckpt_start) * 1000
                checkpoint_times.append(ckpt_time)
                self.checkpoint_count += 1
                context.current_window += 1

                print(f"  Token {token_idx + 1}: Checkpoint #{self.checkpoint_count} "
                      f"written in {ckpt_time:.3f}ms")

        # Force commit any remaining tokens
        if self.checkpoint_mgr and context.tokens_processed % self.window_size != 0:
            expert_locs = {}
            for layer in range(self.num_layers):
                for exp in range(self.num_experts):
                    expert_locs[(layer, exp)] = 0 if exp < 32 else 1
            hot_set = [(layer, exp) for layer in range(self.num_layers)
                       for exp in range(32)]

            ckpt_start = time.time()
            result = self.checkpoint_mgr.force_commit(expert_locs, hot_set)
            ckpt_time = (time.time() - ckpt_start) * 1000

            if result is not None:
                checkpoint_times.append(ckpt_time)
                self.checkpoint_count += 1
                print(f"  Final checkpoint #{self.checkpoint_count} (partial) "
                      f"written in {ckpt_time:.3f}ms")

        elapsed = (time.time() - start) * 1000

        print(f"\nInference complete:")
        print(f"  Tokens processed: {context.tokens_processed}")
        print(f"  Checkpoints created: {self.checkpoint_count}")
        print(f"  Total time: {elapsed:.2f}ms")
        if checkpoint_times:
            print(f"  Avg checkpoint time: {sum(checkpoint_times)/len(checkpoint_times):.3f}ms")
            print(f"  Max checkpoint time: {max(checkpoint_times):.3f}ms")

        return context

    def _simulate_expert_routing(self, token_idx: int, layer_id: int) -> int:
        """Simulate expert routing (in practice, from gating network)."""
        base_expert = (token_idx * 7 + layer_id * 13) % self.num_experts
        if random.random() < 0.3:
            base_expert = (base_expert + random.randint(1, 10)) % self.num_experts
        return base_expert

    def simulate_failure(self, sequence_id: int):
        """Simulate a node/GPU failure."""
        print("\n" + "="*70)
        print("PHASE 3: Simulated Failure")
        print("="*70)

        context = self.contexts.get(sequence_id)
        if not context:
            print(f"No context found for sequence {sequence_id}")
            return

        print(f"\n!!! SIMULATED FAILURE !!!")
        print(f"  Sequence: {sequence_id}")
        print(f"  Tokens processed before failure: {context.tokens_processed}")
        print(f"  Expert routing decisions recorded: {len(context.expert_routing_history)}")

        if self.checkpoint_mgr:
            print(f"  Checkpoints available: {self.checkpoint_mgr.checkpoint_count()}")
            print(f"  Total checkpoint size: {self.checkpoint_mgr.total_checkpoint_size() / 1024:.1f} KB")

        # Clear inference context (simulating memory loss)
        del self.contexts[sequence_id]
        print(f"\n  [Inference state cleared - simulating memory loss]")

    def fast_recovery(self, sequence_id: int) -> Optional[Dict]:
        """Perform fast recovery from CXL checkpoint."""
        print("\n" + "="*70)
        print("PHASE 4: Fast Recovery from CXL Checkpoint")
        print("="*70)

        if not self.checkpoint_mgr:
            print("\n(Simulation mode - no actual recovery)")
            return None

        print("\nReading checkpoint from CXL memory...")

        start = time.time()
        try:
            result = self.checkpoint_mgr.fast_recovery(gpu_ptr=None)
            recovery_time = (time.time() - start) * 1000

            print(f"\n*** RECOVERY COMPLETE ***")
            print(f"  Checkpoint ID: {result['checkpoint_id']}")
            print(f"  Window start: {result['window_start']}")
            print(f"  Window length: {result['window_len']}")
            print(f"  Replay instructions: {len(result['replay_instructions'])}")
            print(f"  Expert locations restored: {len(result['expert_locations'])}")
            print(f"  Hot set size: {len(result['hot_set'])}")
            print(f"  Recovery time (Rust): {result['recovery_time_us']}us")
            print(f"  Recovery time (Python): {recovery_time:.3f}ms")

            # Restore inference context
            self.contexts[sequence_id] = InferenceContext(
                sequence_id=sequence_id,
                tokens_processed=result['window_start'] + result['window_len'],
                current_window=result['window_start'] // self.window_size,
                expert_routing_history=[]
            )

            # Replay routing decisions
            print(f"\nReplaying {len(result['replay_instructions'])} routing decisions...")
            for instr in result['replay_instructions']:
                # In practice, this would update the inference engine's internal state
                pass

            return result

        except Exception as e:
            print(f"\nRecovery failed: {e}")
            return None

    def compare_with_traditional(self, context: InferenceContext):
        """Compare recovery times with traditional approaches."""
        print("\n" + "="*70)
        print("COMPARISON: CXL P2P vs Traditional Recovery")
        print("="*70)

        # Traditional approach timing estimates
        expert_size_gb = 0.256  # 256MB per expert
        total_experts = self.num_experts * self.num_layers
        pcie_bandwidth_gbps = 32  # PCIe Gen4 x16

        # Traditional: reload all experts from storage
        traditional_expert_load_time = (expert_size_gb * total_experts) / pcie_bandwidth_gbps

        # Traditional: re-prefill KV cache
        traditional_kv_prefill_time = context.tokens_processed * 0.05  # ~50ms per token prefill

        traditional_total = traditional_expert_load_time + traditional_kv_prefill_time

        print(f"\nTraditional (DejaVu-style) Recovery:")
        print(f"  Expert weight reload: {traditional_expert_load_time:.1f}s")
        print(f"  KV cache re-prefill: {traditional_kv_prefill_time:.1f}s")
        print(f"  Total: {traditional_total:.1f}s ({traditional_total/60:.1f} minutes)")

        if self.checkpoint_mgr:
            metrics = self.checkpoint_mgr.get_metrics()
            cxl_recovery_time = metrics.get('last_recovery_time_us', 0) / 1000  # Convert to ms

            print(f"\nCXL P2P Recovery:")
            print(f"  Total: {cxl_recovery_time:.3f}ms")

            if cxl_recovery_time > 0:
                speedup = (traditional_total * 1000) / cxl_recovery_time
                print(f"\n  SPEEDUP: {speedup:.0f}x faster!")
        else:
            print("\n(Simulation mode - no actual metrics)")

    def show_metrics(self):
        """Display checkpoint manager metrics."""
        print("\n" + "="*70)
        print("Checkpoint Manager Metrics")
        print("="*70)

        if self.checkpoint_mgr:
            metrics = self.checkpoint_mgr.get_metrics()

            print(f"\nCheckpoint Statistics:")
            print(f"  Checkpoints written: {metrics['checkpoints_written']}")
            print(f"  Checkpoints read: {metrics['checkpoints_read']}")
            print(f"  Bytes written: {metrics['bytes_written']:,}")
            print(f"  Bytes read: {metrics['bytes_read']:,}")

            print(f"\nPerformance:")
            print(f"  Avg write latency: {metrics['avg_write_latency_us']:.1f}us")
            print(f"  Avg read latency: {metrics['avg_read_latency_us']:.1f}us")
            print(f"  Write bandwidth: {metrics['write_bandwidth_gbps']:.2f} GB/s")

            print(f"\nRecovery:")
            print(f"  Recovery count: {metrics['recovery_count']}")
            print(f"  Last recovery time: {metrics['last_recovery_time_us']}us")
        else:
            print("\n(Simulation mode - no metrics available)")

        if self.expert_mgr:
            exp_metrics = self.expert_mgr.get_metrics()
            print(f"\nExpert Manager:")
            print(f"  Cache hits: {exp_metrics['cache_hits']}")
            print(f"  Cache misses: {exp_metrics['cache_misses']}")
            print(f"  Hit rate: {exp_metrics['cache_hit_rate']*100:.1f}%")


def main():
    print("="*70)
    print("CXL P2P Checkpoint/Restore Demo")
    print("="*70)
    print()
    print("This demo shows how CXL memory with P2P DMA enables fast")
    print("checkpoint/restore for MoE models, achieving sub-millisecond")
    print("recovery compared to minutes with traditional approaches.")
    print()
    print("Key features demonstrated:")
    print("  1. Token-to-expert mapping recording during inference")
    print("  2. Windowed checkpointing to CXL memory via P2P DMA")
    print("  3. Fast recovery by reading from CXL and replaying decisions")
    print("  4. Comparison with traditional checkpoint approaches")

    # Initialize demo
    demo = CxlP2pCheckpointDemo(
        num_experts=128,
        num_layers=32,
        window_size=16,
        buffer_size_mb=64  # Smaller buffer for demo
    )

    # Phase 1: Setup
    demo.register_experts()

    # Phase 2: Inference with checkpointing
    context = demo.simulate_inference(sequence_id=42, num_tokens=64)

    # Phase 3: Simulate failure
    demo.simulate_failure(sequence_id=42)

    # Phase 4: Fast recovery
    result = demo.fast_recovery(sequence_id=42)

    # Compare with traditional
    if result:
        demo.compare_with_traditional(context)

    # Show final metrics
    demo.show_metrics()

    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)

    if HAS_CXL_BINDINGS and demo.checkpoint_mgr:
        metrics = demo.checkpoint_mgr.get_metrics()
        recovery_us = metrics.get('last_recovery_time_us', 0)
        if recovery_us > 0:
            print(f"\nRecovery achieved in {recovery_us}us ({recovery_us/1000:.3f}ms)")
            if recovery_us < 1000000:  # < 1 second
                print("\n*** TARGET MET: Sub-second recovery achieved! ***")
    else:
        print("\n(Run with CXL bindings for actual performance measurements)")


if __name__ == "__main__":
    main()
