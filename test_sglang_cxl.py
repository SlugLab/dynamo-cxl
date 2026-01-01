#!/usr/bin/env python3
"""
Test script for SGLang with CXL Checkpoint.

This connects to the running SGLang backend and sends test requests.
"""

import asyncio
import sys
import os

# Add components to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "components/src"))

from dynamo.runtime import DistributedRuntime


async def test_generation():
    """Test generation with CXL checkpoint backend."""
    print("Connecting to Dynamo runtime...")

    # Create runtime (connects to etcd)
    runtime = DistributedRuntime.detached()

    # Get the backend client
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    print(f"Namespace: {namespace}")

    client = await (
        runtime.namespace(namespace)
        .component("backend")
        .endpoint("generate")
        .client()
    )

    print("Waiting for backend instances...")
    instances = await client.wait_for_instances()
    print(f"Found {len(instances)} backend instance(s)")

    # Prepare request
    request = {
        "token_ids": [1, 2, 3, 4, 5],  # Dummy tokens
        "sampling_options": {
            "temperature": 0.7,
        },
        "stop_conditions": {
            "max_tokens": 20,
        },
    }

    print("\nSending generation request...")
    print(f"Input tokens: {request['token_ids']}")

    # Send request and collect response
    output_tokens = []
    async for response in await client.generate(request):
        data = response.data()
        if data:
            tokens = data.get("token_ids", [])
            output_tokens.extend(tokens)
            if data.get("finish_reason"):
                print(f"Finish reason: {data['finish_reason']}")
                break

    print(f"Output tokens: {output_tokens}")
    print(f"Total tokens generated: {len(output_tokens)}")

    print("\n[SUCCESS] Generation with CXL checkpoint completed!")


def main():
    print("=" * 60)
    print("SGLang CXL Checkpoint Test")
    print("=" * 60)

    try:
        asyncio.run(test_generation())
        return 0
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
