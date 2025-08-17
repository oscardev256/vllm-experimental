# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for checking endpoint readiness."""

import asyncio
import time

import aiohttp
from tqdm.asyncio import tqdm

from .endpoint_request_func import RequestFuncInput, RequestFuncOutput


async def wait_for_endpoint(
    request_func,
    test_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    timeout_seconds: int = 600,
    retry_interval: int = 5,
) -> RequestFuncOutput:
    """
    Wait for an endpoint to become available before starting benchmarks.
    
    Args:
        request_func: The async request function to call
        test_input: The RequestFuncInput to test with
        timeout_seconds: Maximum time to wait in seconds (default: 10 minutes)
        retry_interval: Time between retries in seconds (default: 5 seconds)
        
    Returns:
        RequestFuncOutput: The successful response
        
    Raises:
        ValueError: If the endpoint doesn't become available within the timeout
    """
    deadline = time.perf_counter() + timeout_seconds
    output = RequestFuncOutput(success=False)
    print(f"Waiting for endpoint to become up in {timeout_seconds} seconds")
    
    with tqdm(
        total=timeout_seconds, 
        bar_format="{desc} |{bar}| {elapsed} elapsed, {remaining} remaining",
        unit="s",
    ) as pbar:

        while True:            
            # update progress bar
            remaining = deadline - time.perf_counter()
            elapsed = timeout_seconds - remaining
            update_amount = min(elapsed - pbar.n, timeout_seconds - pbar.n)
            pbar.update(update_amount)
            pbar.refresh()
            if remaining <= 0:
                pbar.close()
                break

            # ping the endpoint using request_func
            try:
                print(f"[DEBUG] Attempting readiness check with test_input:")
                print(f"[DEBUG]   prompt: {test_input.prompt[:100]}...")
                print(f"[DEBUG]   model: {test_input.model}")
                print(f"[DEBUG]   api_url: {test_input.api_url}")
                print(f"[DEBUG]   multi_modal_content: {test_input.multi_modal_content}")
                if hasattr(test_input, 'multi_modal_content') and test_input.multi_modal_content:
                    if isinstance(test_input.multi_modal_content, dict):
                        print(f"[DEBUG]   multi_modal_content keys: {list(test_input.multi_modal_content.keys())}")
                        for key, value in test_input.multi_modal_content.items():
                            if key == "audio" and isinstance(value, tuple):
                                print(f"[DEBUG]     {key}: (array_shape={value[0].shape if hasattr(value[0], 'shape') else type(value[0])}, sr={value[1]})")
                            else:
                                print(f"[DEBUG]     {key}: {type(value)}")
                    else:
                        print(f"[DEBUG]   multi_modal_content is a list with {len(test_input.multi_modal_content)} items")
                
                output = await request_func(
                    request_func_input=test_input, session=session)
                
                print(f"[DEBUG] Readiness check response:")
                print(f"[DEBUG]   success: {output.success}")
                print(f"[DEBUG]   error_code: {getattr(output, 'error_code', None)}")
                print(f"[DEBUG]   error_text: {getattr(output, 'error_text', None)}")
                if hasattr(output, 'generated_text'):
                    print(f"[DEBUG]   generated_text: {output.generated_text[:200] if output.generated_text else None}...")
                
                if output.success:
                    pbar.close()
                    return output
                else:
                    print(f"[DEBUG] Readiness check failed, will retry...")
                    
            except aiohttp.ClientConnectorError as e:
                print(f"[DEBUG] Connection error during readiness check: {e}")
                pass
            except Exception as e:
                print(f"[DEBUG] Unexpected error during readiness check: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            
            # retry after a delay
            sleep_duration = min(retry_interval, remaining)
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)
    
    return output
