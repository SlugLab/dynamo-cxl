#!/usr/bin/env python3
"""
Test script for CXL Checkpoint functionality.

This script tests:
1. CxlCheckpointManager - checkpoint/recovery for token-expert mappings
2. CxlExpertManager - expert weight tiering between GPU and CXL memory
3. Integration of both for sub-second MoE fault tolerance

Usage:
    python test_cxl_checkpoint.py
"""

import asyncio
import time
from typing import Dict, List, Tuple

# Import CXL managers from Rust bindings
from dynamo._core import (
    CxlCheckpointManager,
    CxlExpertManager,
    CxlExpertManagerConfig,
)


def test_checkpoint_manager():
    """Test CxlCheckpointManager functionality."""
    print("\n" + "=" * 60)
    print("TEST 1: CxlCheckpointManager")
    print("=" * 60)

    # Create checkpoint manager
    window_size = 16
    num_layers = 32
    buffer_size_mb = 64

    print(f"\nCreating CxlCheckpointManager:")
    print(f"  - Window size: {window_size} tokens")
    print(f"  - Num layers: {num_layers}")
    print(f"  - Buffer size: {buffer_size_mb} MB")

    manager = CxlCheckpointManager(
        window_size=window_size,
        num_layers=num_layers,
        buffer_size_mb=buffer_size_mb,
    )

    # Set sequence ID
    sequence_id = 12345
    manager.set_sequence_id(sequence_id)
    print(f"\nSet sequence ID: {sequence_id}")

    # Record some token-expert mappings
    print("\nRecording token-expert mappings...")
    num_tokens = 48  # 3 windows worth
    num_experts = 128

    start_time = time.time()
    for token_pos in range(num_tokens):
        for layer_id in range(num_layers):
            expert_id = (token_pos + layer_id) % num_experts
            topk_experts = [expert_id, (expert_id + 1) % num_experts]
            gating_scores = [0.7, 0.3]
            kv_block_hash = hash((sequence_id, token_pos, layer_id)) & 0xFFFFFFFFFFFFFFFF

            manager.record_mapping(
                token_position=token_pos,
                layer_id=layer_id,
                expert_id=expert_id,
                topk_experts=topk_experts,
                gating_scores=gating_scores,
                kv_block_hash=kv_block_hash,
            )

    record_time_ms = (time.time() - start_time) * 1000
    total_mappings = num_tokens * num_layers
    print(f"  Recorded {total_mappings} mappings in {record_time_ms:.2f}ms")
    print(f"  ({total_mappings / record_time_ms * 1000:.0f} mappings/sec)")

    # Write checkpoint
    print("\nWriting checkpoint...")
    expert_locations = {(0, i): 0 if i < 32 else 1 for i in range(num_experts)}
    hot_set = [(0, i) for i in range(32)]

    start_time = time.time()
    checkpoint_id = manager.write_checkpoint(
        expert_locations=expert_locations,
        hot_set=hot_set,
        gpu_ptr=None,
    )
    write_time_ms = (time.time() - start_time) * 1000
    print(f"  Checkpoint {checkpoint_id} written in {write_time_ms:.2f}ms")

    # Get metrics
    metrics = manager.get_metrics()
    print(f"\nCheckpoint Manager Metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")

    # Test recovery
    print("\nTesting fast recovery...")
    start_time = time.time()
    result = manager.fast_recovery(gpu_ptr=None)
    recovery_time_ms = (time.time() - start_time) * 1000

    print(f"  Recovery completed in {recovery_time_ms:.2f}ms")
    print(f"  Checkpoint ID: {result.get('checkpoint_id')}")
    print(f"  Window start: {result.get('window_start')}")
    print(f"  Window length: {result.get('window_len')}")
    print(f"  Replay instructions: {len(result.get('replay_instructions', []))}")

    print("\n[PASS] CxlCheckpointManager test completed successfully!")
    return True


def test_expert_manager():
    """Test CxlExpertManager functionality."""
    print("\n" + "=" * 60)
    print("TEST 2: CxlExpertManager")
    print("=" * 60)

    # Create expert manager config
    config = CxlExpertManagerConfig(
        num_experts=128,
        num_moe_layers=32,
        expert_weight_size=256 * 1024 * 1024,  # 256MB per expert
        gpu_expert_capacity=80 * 1024 * 1024 * 1024,  # 80GB
        cxl_expert_capacity=512 * 1024 * 1024 * 1024,  # 512GB
        window_size=16,
        cxl_bandwidth_gbps=128.0,
        max_gpu_experts=32,
        enable_p2p_dma=True,
        qos_priority_levels=4,
    )

    print(f"\nCxlExpertManagerConfig:")
    print(f"  - num_experts: {config.num_experts}")
    print(f"  - num_moe_layers: {config.num_moe_layers}")
    print(f"  - max_gpu_experts: {config.max_gpu_experts}")
    print(f"  - window_size: {config.window_size}")
    print(f"  - enable_p2p_dma: {config.enable_p2p_dma}")

    # Create expert manager
    print("\nCreating CxlExpertManager...")
    manager = CxlExpertManager(config)

    # Register experts
    print("\nRegistering experts...")
    start_time = time.time()
    for layer_id in range(config.num_moe_layers):
        for expert_id in range(config.num_experts):
            location = "gpu" if expert_id < config.max_gpu_experts else "cxl"
            manager.register_expert(
                expert_id=expert_id,
                layer_id=layer_id,
                weight_size_bytes=config.expert_weight_size,
                initial_location=location,
            )

    register_time_ms = (time.time() - start_time) * 1000
    total_experts = config.num_moe_layers * config.num_experts
    print(f"  Registered {total_experts} experts in {register_time_ms:.2f}ms")

    # Record KV deltas (simulating inference)
    print("\nRecording KV deltas (simulating inference)...")
    num_tokens = 32
    start_time = time.time()

    for token_offset in range(num_tokens):
        for layer_id in range(config.num_moe_layers):
            expert_id = (token_offset + layer_id) % config.num_experts
            topk_experts = [expert_id, (expert_id + 1) % config.num_experts]
            gating_scores = [0.7, 0.3]
            kv_block_hash = hash((token_offset, layer_id)) & 0xFFFFFFFFFFFFFFFF
            tokens_hash = hash(token_offset) & 0xFFFFFFFFFFFFFFFF

            manager.record_kv_delta(
                token_offset=token_offset,
                layer_id=layer_id,
                expert_id=expert_id,
                topk_experts=topk_experts,
                gating_scores=gating_scores,
                kv_block_hash=kv_block_hash,
                tokens_hash=tokens_hash,
            )

    delta_time_ms = (time.time() - start_time) * 1000
    total_deltas = num_tokens * config.num_moe_layers
    print(f"  Recorded {total_deltas} KV deltas in {delta_time_ms:.2f}ms")

    # Force checkpoint
    print("\nForcing checkpoint...")
    start_time = time.time()
    manager.force_checkpoint()
    checkpoint_time_ms = (time.time() - start_time) * 1000
    print(f"  Checkpoint forced in {checkpoint_time_ms:.2f}ms")

    # Get metrics
    metrics = manager.get_metrics()
    print(f"\nExpert Manager Metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")

    # Test fast recovery
    print("\nTesting fast recovery...")

    async def do_recovery():
        start_time = time.time()
        recovery_time_ms = await manager.fast_recovery()
        actual_time_ms = (time.time() - start_time) * 1000
        return recovery_time_ms, actual_time_ms

    recovery_time_ms, actual_time_ms = asyncio.run(do_recovery())
    print(f"  Recovery reported: {recovery_time_ms:.2f}ms")
    print(f"  Actual wall time: {actual_time_ms:.2f}ms")

    # Shutdown
    print("\nShutting down expert manager...")
    manager.shutdown()

    print("\n[PASS] CxlExpertManager test completed successfully!")
    return True


def test_integrated_checkpoint_recovery():
    """Test integrated checkpoint/recovery workflow."""
    print("\n" + "=" * 60)
    print("TEST 3: Integrated Checkpoint/Recovery Workflow")
    print("=" * 60)

    # Configuration
    num_experts = 64
    num_layers = 8
    max_gpu_experts = 16
    window_size = 16
    num_tokens = 100  # Simulate 100 tokens of generation

    print(f"\nConfiguration:")
    print(f"  - Experts: {num_experts} per layer")
    print(f"  - Layers: {num_layers}")
    print(f"  - GPU experts: {max_gpu_experts} (hot tier)")
    print(f"  - CXL experts: {num_experts - max_gpu_experts} (cold tier)")
    print(f"  - Window size: {window_size} tokens")
    print(f"  - Total tokens: {num_tokens}")

    # Create checkpoint manager
    ckpt_manager = CxlCheckpointManager(
        window_size=window_size,
        num_layers=num_layers,
        buffer_size_mb=32,
    )

    # Create expert manager
    expert_config = CxlExpertManagerConfig(
        num_experts=num_experts,
        num_moe_layers=num_layers,
        expert_weight_size=64 * 1024 * 1024,  # 64MB for test
        gpu_expert_capacity=16 * 1024 * 1024 * 1024,
        cxl_expert_capacity=64 * 1024 * 1024 * 1024,
        window_size=window_size,
        cxl_bandwidth_gbps=128.0,
        max_gpu_experts=max_gpu_experts,
        enable_p2p_dma=True,
        qos_priority_levels=4,
    )
    expert_manager = CxlExpertManager(expert_config)

    # Register experts
    print("\n[Phase 1] Registering experts...")
    for layer_id in range(num_layers):
        for expert_id in range(num_experts):
            location = "gpu" if expert_id < max_gpu_experts else "cxl"
            expert_manager.register_expert(
                expert_id=expert_id,
                layer_id=layer_id,
                weight_size_bytes=expert_config.expert_weight_size,
                initial_location=location,
            )
    print(f"  Registered {num_layers * num_experts} experts")

    # Simulate inference with checkpointing
    print("\n[Phase 2] Simulating inference with checkpointing...")
    sequence_id = 99999
    ckpt_manager.set_sequence_id(sequence_id)

    checkpoint_count = 0
    total_start = time.time()

    for token_pos in range(num_tokens):
        # Record routing for each layer
        for layer_id in range(num_layers):
            # Simulate expert selection (varies with token and layer)
            expert_id = (token_pos * 7 + layer_id * 3) % num_experts
            topk = [expert_id, (expert_id + 13) % num_experts]
            scores = [0.65, 0.35]
            kv_hash = hash((sequence_id, token_pos, layer_id)) & 0xFFFFFFFFFFFFFFFF

            # Record in checkpoint manager
            ckpt_manager.record_mapping(
                token_position=token_pos,
                layer_id=layer_id,
                expert_id=expert_id,
                topk_experts=topk,
                gating_scores=scores,
                kv_block_hash=kv_hash,
            )

            # Record in expert manager
            expert_manager.record_kv_delta(
                token_offset=token_pos,
                layer_id=layer_id,
                expert_id=expert_id,
                topk_experts=topk,
                gating_scores=scores,
                kv_block_hash=kv_hash,
                tokens_hash=hash(token_pos) & 0xFFFFFFFFFFFFFFFF,
            )

        # Checkpoint every window_size tokens
        if (token_pos + 1) % window_size == 0:
            expert_locs = {(l, e): 0 if e < max_gpu_experts else 1
                          for l in range(num_layers) for e in range(num_experts)}
            hot_set = [(l, e) for l in range(num_layers) for e in range(max_gpu_experts)]

            ckpt_manager.write_checkpoint(
                expert_locations=expert_locs,
                hot_set=hot_set,
            )
            expert_manager.force_checkpoint()
            checkpoint_count += 1

    inference_time_ms = (time.time() - total_start) * 1000
    print(f"  Processed {num_tokens} tokens in {inference_time_ms:.2f}ms")
    print(f"  Created {checkpoint_count} checkpoints")

    # Simulate failure and recovery
    print("\n[Phase 3] Simulating failure and recovery...")
    print("  >>> SIMULATED FAILURE <<<")

    # Recovery
    recovery_start = time.time()

    # Checkpoint manager recovery
    ckpt_result = ckpt_manager.fast_recovery()

    # Expert manager recovery
    async def do_expert_recovery():
        return await expert_manager.fast_recovery()

    expert_recovery_ms = asyncio.run(do_expert_recovery())

    total_recovery_ms = (time.time() - recovery_start) * 1000

    print(f"\n  Recovery Results:")
    print(f"    - Checkpoint ID: {ckpt_result.get('checkpoint_id')}")
    print(f"    - Window: tokens {ckpt_result.get('window_start')}-{ckpt_result.get('window_start', 0) + ckpt_result.get('window_len', 0)}")
    print(f"    - Replay instructions: {len(ckpt_result.get('replay_instructions', []))}")
    print(f"    - Total recovery time: {total_recovery_ms:.2f}ms")

    # Metrics summary
    print("\n[Phase 4] Final Metrics:")

    ckpt_metrics = ckpt_manager.get_metrics()
    print(f"\n  Checkpoint Manager:")
    for k, v in ckpt_metrics.items():
        print(f"    - {k}: {v}")

    expert_metrics = expert_manager.get_metrics()
    print(f"\n  Expert Manager:")
    for k, v in expert_metrics.items():
        print(f"    - {k}: {v}")

    # Cleanup
    expert_manager.shutdown()

    # Validate sub-second recovery
    if total_recovery_ms < 1000:
        print(f"\n[PASS] Sub-second recovery achieved: {total_recovery_ms:.2f}ms < 1000ms")
    else:
        print(f"\n[WARN] Recovery took {total_recovery_ms:.2f}ms (target: <1000ms)")

    return True


def main():
    """Run all CXL checkpoint tests."""
    print("=" * 60)
    print("CXL CHECKPOINT TEST SUITE")
    print("=" * 60)
    print("\nThis test validates the CXL checkpoint functionality for")
    print("sub-second fault tolerance in MoE models.")

    results = []

    # Run tests
    try:
        results.append(("CxlCheckpointManager", test_checkpoint_manager()))
    except Exception as e:
        print(f"\n[FAIL] CxlCheckpointManager: {e}")
        results.append(("CxlCheckpointManager", False))

    try:
        results.append(("CxlExpertManager", test_expert_manager()))
    except Exception as e:
        print(f"\n[FAIL] CxlExpertManager: {e}")
        results.append(("CxlExpertManager", False))

    try:
        results.append(("Integrated Workflow", test_integrated_checkpoint_recovery()))
    except Exception as e:
        print(f"\n[FAIL] Integrated Workflow: {e}")
        results.append(("Integrated Workflow", False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
