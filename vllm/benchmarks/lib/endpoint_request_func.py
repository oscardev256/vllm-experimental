# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The request function for API endpoints."""

import io
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
from tqdm.asyncio import tqdm

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    """The input for the request function."""
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: Optional[str] = None
    logprobs: Optional[int] = None
    extra_body: Optional[dict] = None
    multi_modal_content: Optional[dict | list[dict]] = None
    ignore_eos: bool = False
    language: Optional[str] = None


@dataclass
class RequestFuncOutput:
    """The output of the request function including metrics."""
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(
        default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    """The async request function for the OpenAI Completions API.

    Args:
        request_func_input: The input for the request function.
        pbar: The progress bar to display the progress.

    Returns:
        The output of the request function.
    """
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."

    payload = {
        "model": request_func_input.model_name \
            if request_func_input.model_name else request_func_input.model,
        "prompt": request_func_input.prompt,
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": request_func_input.output_len,
        "logprobs": request_func_input.logprobs,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }
    if request_func_input.ignore_eos:
        payload["ignore_eos"] = request_func_input.ignore_eos
    if request_func_input.extra_body:
        payload.update(request_func_input.extra_body)
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
    }

    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len

    generated_text = ""
    st = time.perf_counter()
    most_recent_timestamp = st
    try:
        async with session.post(url=api_url, json=payload,
                                headers=headers) as response:
            if response.status == 200:
                first_chunk_received = False
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk_bytes = chunk_bytes.decode("utf-8")
                    # NOTE: SSE comments (often used as pings) start with
                    # a colon. These are not JSON data payload and should
                    # be skipped.
                    if chunk_bytes.startswith(":"):
                        continue

                    chunk = chunk_bytes.removeprefix("data: ")

                    if chunk != "[DONE]":
                        data = json.loads(chunk)

                        # NOTE: Some completion API might have a last
                        # usage summary response without a token so we
                        # want to check a token was generated
                        if choices := data.get("choices"):
                            # Note that text could be empty here
                            # e.g. for special tokens
                            text = choices[0].get("text")
                            timestamp = time.perf_counter()
                            # First token
                            if not first_chunk_received:
                                first_chunk_received = True
                                ttft = time.perf_counter() - st
                                output.ttft = ttft

                            # Decoding phase
                            else:
                                output.itl.append(timestamp -
                                                    most_recent_timestamp)

                            most_recent_timestamp = timestamp
                            generated_text += text or ""
                        elif usage := data.get("usage"):
                            output.output_tokens = usage.get(
                                "completion_tokens")
                if first_chunk_received:
                    output.success = True
                else:
                    output.success = False
                    output.error = (
                        "Never received a valid chunk to calculate TTFT."
                        "This response will be marked as failed!")
                output.generated_text = generated_text
                output.latency = most_recent_timestamp - st
            else:
                output.error = response.reason or ""
                output.success = False
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(("chat/completions", "profile")), (
        "OpenAI Chat Completions API URL must end with 'chat/completions'.")

    print(f"[DEBUG] Building OpenAI chat request:")
    print(f"[DEBUG]   api_url: {api_url}")
    print(f"[DEBUG]   prompt type: {type(request_func_input.prompt)}")
    print(f"[DEBUG]   multi_modal_content: {request_func_input.multi_modal_content}")

    # Check if prompt is already in chat format (list with role/content structure)
    if isinstance(request_func_input.prompt, list):
        # Prompt is already in chat format, extract the content
        messages = request_func_input.prompt
        print(f"[DEBUG]   Using existing chat format with {len(messages)} messages")
    else:
        # Prompt is a string, create chat format
        content = [{"type": "text", "text": request_func_input.prompt}]
        print(f"[DEBUG]   prompt: {request_func_input.prompt[:100]}...")
        messages = None
    # Handle multimodal content only if prompt is not already in chat format
    if not messages and request_func_input.multi_modal_content:
        mm_content = request_func_input.multi_modal_content
        print(f"[DEBUG]   mm_content type: {type(mm_content)}")
        if isinstance(mm_content, dict):
            print(f"[DEBUG]   mm_content keys: {list(mm_content.keys())}")
            for key, value in mm_content.items():
                if key == "audio" and isinstance(value, tuple):
                    print(f"[DEBUG]     {key}: (array_shape={value[0].shape if hasattr(value[0], 'shape') else type(value[0])}, sr={value[1]})")
                else:
                    print(f"[DEBUG]     {key}: {type(value)}")
        
        if isinstance(mm_content, list):
            content.extend(mm_content)
        elif isinstance(mm_content, dict):
            content.append(mm_content)
        else:
            raise TypeError(
                "multi_modal_content must be a dict or list[dict] "
                "for openai-chat"
            )
    
    if messages:
        # Use existing chat format
        payload_messages = messages
        print(f"[DEBUG]   Using chat format with {len(messages)} messages")
    else:
        # Create chat format from content
        print(f"[DEBUG]   final content length: {len(content)}")
        for i, item in enumerate(content):
            if isinstance(item, dict):
                item_type = item.get("type", "unknown")
                print(f"[DEBUG]     content[{i}]: type={item_type}")
                if item_type == "audio" and isinstance(item.get("audio"), tuple):
                    audio_tuple = item["audio"]
                    print(f"[DEBUG]       audio: (array_shape={audio_tuple[0].shape if hasattr(audio_tuple[0], 'shape') else type(audio_tuple[0])}, sr={audio_tuple[1]})")
            else:
                print(f"[DEBUG]     content[{i}]: {type(item)}")
        payload_messages = [
            {
                "role": "user",
                "content": content
            },
        ]
    
    payload = {
        "model":
        request_func_input.model_name
        if request_func_input.model_name else request_func_input.model,
        "messages": payload_messages,
        "temperature":
        0.0,
        "max_completion_tokens":
        request_func_input.output_len,
        "stream":
        True,
        "stream_options": {
            "include_usage": True,
        },
    }
    if request_func_input.ignore_eos:
        payload["ignore_eos"] = request_func_input.ignore_eos
    if request_func_input.extra_body:
        payload.update(request_func_input.extra_body)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }

    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len

    generated_text = ""
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    
    print(f"[DEBUG] Sending request to {api_url}")
    print(f"[DEBUG] Payload: {json.dumps(payload, indent=2)}")
    
    try:
        async with session.post(url=api_url, json=payload,
                                headers=headers) as response:
            print(f"[DEBUG] Response status: {response.status}")
            if response.status != 200:
                response_text = await response.text()
                print(f"[DEBUG] Error response: {response_text}")
            
            if response.status == 200:
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk_bytes = chunk_bytes.decode("utf-8")
                    # NOTE: SSE comments (often used as pings) start with
                    # a colon. These are not JSON data payload and should
                    # be skipped.
                    if chunk_bytes.startswith(":"):
                        continue

                    chunk = chunk_bytes.removeprefix("data: ")

                    if chunk != "[DONE]":
                        timestamp = time.perf_counter()
                        data = json.loads(chunk)

                        if choices := data.get("choices"):
                            content = choices[0]["delta"].get("content")
                            # First token
                            if ttft == 0.0:
                                ttft = timestamp - st
                                output.ttft = ttft

                            # Decoding phase
                            else:
                                output.itl.append(timestamp -
                                                    most_recent_timestamp)

                            generated_text += content or ""
                        elif usage := data.get("usage"):
                            output.output_tokens = usage.get(
                                "completion_tokens")

                        most_recent_timestamp = timestamp

                output.generated_text = generated_text
                output.success = True
                output.latency = most_recent_timestamp - st
            else:
                output.error = response.reason or ""
                output.success = False
    except Exception:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_audio(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    # Lazy import without PlaceholderModule to avoid vllm dep.
    import soundfile

    api_url = request_func_input.api_url
    assert api_url.endswith(("transcriptions", "translations")), (
        "OpenAI Chat Completions API URL must end with 'transcriptions' ")
    "or `translations`."

    content = [{"type": "text", "text": request_func_input.prompt}]
    payload = {
        "model":
        request_func_input.model_name
        if request_func_input.model_name else request_func_input.model,
        "temperature":
        0.0,
        "max_completion_tokens":
        request_func_input.output_len,
        "stream":
        True,
        "language":
        "en",
        # Flattened due to multipart/form-data
        "stream_include_usage":
        True,
        "stream_continuous_usage_stats":
        True,
    }
    if request_func_input.extra_body:
        payload.update(request_func_input.extra_body)
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }

    # Send audio file
    def to_bytes(y, sr):
        buffer = io.BytesIO()
        soundfile.write(buffer, y, sr, format="WAV")
        buffer.seek(0)
        return buffer

    mm_audio = request_func_input.multi_modal_content
    if not isinstance(mm_audio, dict) or "audio" not in mm_audio:
        raise TypeError("multi_modal_content must be a dict containing 'audio'")
    with to_bytes(*mm_audio["audio"]) as f:
        form = aiohttp.FormData()
        form.add_field("file", f, content_type="audio/wav")
        for key, value in payload.items():
            form.add_field(key, str(value))

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url,
                                    data=form,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data: ")
                        if chunk != "[DONE]":
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                content = choices[0]["delta"].get(
                                    "content")
                                # First token
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(
                                        timestamp - most_recent_timestamp)

                                generated_text += content or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get(
                                    "completion_tokens")

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


# TODO: Add more request functions for different API protocols.
ASYNC_REQUEST_FUNCS = {
    "vllm": async_request_openai_completions,
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
    "openai-audio": async_request_openai_audio,
}

OPENAI_COMPATIBLE_BACKENDS = [
    k for k, v in ASYNC_REQUEST_FUNCS.items()
    if v in (async_request_openai_completions,
             async_request_openai_chat_completions)
]
