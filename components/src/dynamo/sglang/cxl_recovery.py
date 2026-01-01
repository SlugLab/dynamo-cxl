# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CXL Recovery Service for SGLang MoE models.

Provides fault tolerance through sub-second checkpoint/recovery:
- Records expert routing decisions during inference
- Stores checkpoints in CXL memory with P2P DMA
- Replays routing decisions on recovery without re-prefill

Usage:
    # Start with CXL checkpoint enabled
    python -m dynamo.sglang --enable-cxl-checkpoint --model Qwen/Qwen3-30B-A3B

    # Trigger recovery via endpoint
    curl -X POST http://localhost:8000/v1/cxl/recover

    # Get CXL metrics
    curl http://localhost:8000/v1/cxl/metrics
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from dynamo._core import Component, Endpoint


@dataclass
class RecoveryRequest:
    """Request to trigger CXL checkpoint recovery."""

    # Optional GPU pointer for P2P DMA recovery
    gpu_ptr: Optional[int] = None

    # Specific checkpoint ID to recover from (None = latest)
    checkpoint_id: Optional[int] = None

    # Whether to return full replay instructions
    return_instructions: bool = False


@dataclass
class RecoveryResponse:
    """Response from CXL checkpoint recovery."""

    success: bool
    checkpoint_id: int
    window_start: int
    window_len: int
    recovery_time_ms: float
    replay_instructions_count: int
    expert_locations_restored: int
    hot_set_size: int
    error: Optional[str] = None


class CxlRecoveryService:
    """
    Service for managing CXL checkpoint recovery.

    This service provides endpoints for:
    - Triggering recovery from checkpoints
    - Getting CXL checkpoint metrics
    - Forcing checkpoint creation
    - Managing expert locations

    Architecture:
        The recovery service maintains a reference to the CXL handlers
        and provides a control plane for checkpoint/recovery operations.

    Recovery Flow:
        1. Client sends recovery request
        2. Service reads latest checkpoint from CXL memory
        3. Routing decisions are replayed (expert selection restored)
        4. Inference can resume from checkpoint without re-prefill
    """

    def __init__(
        self,
        component: Component,
        handler: Any,  # CxlDecodeWorkerHandler or CxlPrefillWorkerHandler
    ):
        self.component = component
        self.handler = handler
        self._recovery_lock = asyncio.Lock()

    async def recover(self, request: Dict) -> Dict:
        """
        Trigger recovery from the latest CXL checkpoint.

        Args:
            request: Dict with optional fields:
                - gpu_ptr: GPU pointer for P2P DMA (optional)
                - checkpoint_id: Specific checkpoint to recover (optional)
                - return_instructions: Include replay instructions (optional)

        Returns:
            Dict with recovery information
        """
        async with self._recovery_lock:
            start_time = time.time()

            try:
                gpu_ptr = request.get("gpu_ptr")
                result = await self.handler.fast_recovery(gpu_ptr=gpu_ptr)

                if "error" in result:
                    return {
                        "success": False,
                        "error": result["error"],
                        "recovery_time_ms": (time.time() - start_time) * 1000,
                    }

                response = {
                    "success": True,
                    "checkpoint_id": result.get("checkpoint_id", 0),
                    "window_start": result.get("window_start", 0),
                    "window_len": result.get("window_len", 0),
                    "recovery_time_ms": (time.time() - start_time) * 1000,
                    "replay_instructions_count": len(result.get("replay_instructions", [])),
                    "expert_locations_restored": len(result.get("expert_locations", {})),
                    "hot_set_size": len(result.get("hot_set", [])),
                }

                if request.get("return_instructions", False):
                    response["replay_instructions"] = result.get("replay_instructions", [])

                logging.info(
                    f"CXL recovery completed: checkpoint {response['checkpoint_id']}, "
                    f"{response['replay_instructions_count']} instructions, "
                    f"{response['recovery_time_ms']:.2f}ms"
                )

                return response

            except Exception as e:
                logging.error(f"CXL recovery failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "recovery_time_ms": (time.time() - start_time) * 1000,
                }

    async def force_checkpoint(self, request: Dict) -> Dict:
        """
        Force creation of a checkpoint.

        Args:
            request: Empty dict (no parameters needed)

        Returns:
            Dict with checkpoint information
        """
        try:
            checkpoint_id = self.handler.force_checkpoint()

            if checkpoint_id is not None:
                return {
                    "success": True,
                    "checkpoint_id": checkpoint_id,
                    "message": "Checkpoint created successfully",
                }
            else:
                return {
                    "success": True,
                    "checkpoint_id": None,
                    "message": "No pending data to checkpoint",
                }

        except Exception as e:
            logging.error(f"Force checkpoint failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def get_metrics(self, request: Dict) -> Dict:
        """
        Get CXL checkpoint and expert manager metrics.

        Args:
            request: Empty dict (no parameters needed)

        Returns:
            Dict with CXL metrics
        """
        try:
            metrics = self.handler.get_cxl_metrics()
            return {
                "success": True,
                **metrics,
            }
        except Exception as e:
            logging.error(f"Failed to get CXL metrics: {e}")
            return {
                "success": False,
                "error": str(e),
            }


async def setup_cxl_recovery_endpoints(
    component: Component,
    handler: Any,
) -> CxlRecoveryService:
    """
    Set up CXL recovery service endpoints.

    This creates endpoints for:
    - /recover: Trigger recovery from checkpoint
    - /checkpoint: Force checkpoint creation
    - /metrics: Get CXL metrics

    Args:
        component: Dynamo component
        handler: CXL-aware handler (CxlDecodeWorkerHandler or CxlPrefillWorkerHandler)

    Returns:
        CxlRecoveryService instance
    """
    service = CxlRecoveryService(component, handler)

    # Create recovery endpoint
    recover_endpoint = component.endpoint("cxl_recover")
    await recover_endpoint.serve_endpoint(
        service.recover,
        graceful_shutdown=True,
    )

    # Create force checkpoint endpoint
    checkpoint_endpoint = component.endpoint("cxl_checkpoint")
    await checkpoint_endpoint.serve_endpoint(
        service.force_checkpoint,
        graceful_shutdown=True,
    )

    # Create metrics endpoint
    metrics_endpoint = component.endpoint("cxl_metrics")
    await metrics_endpoint.serve_endpoint(
        service.get_metrics,
        graceful_shutdown=True,
    )

    logging.info("CXL recovery endpoints configured: cxl_recover, cxl_checkpoint, cxl_metrics")

    return service


# Standalone function for direct recovery without service
async def perform_cxl_recovery(
    handler: Any,
    gpu_ptr: Optional[int] = None,
) -> Dict:
    """
    Perform CXL checkpoint recovery directly.

    This is a convenience function for triggering recovery without
    going through the service endpoints.

    Args:
        handler: CXL-aware handler
        gpu_ptr: Optional GPU pointer for P2P DMA

    Returns:
        Dict with recovery results
    """
    start_time = time.time()

    try:
        result = await handler.fast_recovery(gpu_ptr=gpu_ptr)

        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "recovery_time_ms": (time.time() - start_time) * 1000,
            }

        return {
            "success": True,
            "checkpoint_id": result.get("checkpoint_id", 0),
            "recovery_time_ms": (time.time() - start_time) * 1000,
            "replay_instructions_count": len(result.get("replay_instructions", [])),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "recovery_time_ms": (time.time() - start_time) * 1000,
        }
