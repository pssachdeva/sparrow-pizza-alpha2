"""Shared execution helpers for vignette async and provider-native batch runs."""
from __future__ import annotations

from copy import deepcopy
import json
import os
import time
from pathlib import Path

from anthropic import Anthropic
from google import genai
from litellm import batch_completion
from loguru import logger
from openai import OpenAI

from law_norms_llms.expansion import ExecutionTask
from law_norms_llms.providers import (
    OPENAI_NO_TEMP_MODELS,
    apply_anthropic_reasoning,
    apply_openai_reasoning,
    build_anthropic_messages,
    build_google_contents,
    build_openai_messages,
    extract_json,
)
from law_norms_llms.schemas import get_schema, make_refusal_response


def build_batch_request_id(task_index: int, repeat: int) -> str:
    """Build a short provider-facing batch request identifier."""
    return f"req{task_index:05d}_rep{repeat:03d}"


def sanitize_google_response_schema(value):
    """Remove JSON Schema fields that Gemini batch rejects."""
    unsupported_keys = {"additionalProperties", "title", "default"}
    if isinstance(value, dict):
        return {
            key: sanitize_google_response_schema(subvalue)
            for key, subvalue in value.items()
            if key not in unsupported_keys
        }
    if isinstance(value, list):
        return [sanitize_google_response_schema(item) for item in value]
    return value


def convert_json_schema_to_google_schema(value):
    """Translate a JSON Schema subset into Gemini's response_schema format."""
    if not isinstance(value, dict):
        return value

    type_mapping = {
        "object": "OBJECT",
        "string": "STRING",
        "integer": "INTEGER",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
    }

    schema = sanitize_google_response_schema(value)
    converted: dict[str, object] = {}

    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        converted["type"] = type_mapping.get(schema_type, schema_type.upper())

    if "properties" in schema:
        converted["properties"] = {
            key: convert_json_schema_to_google_schema(subschema)
            for key, subschema in schema["properties"].items()
        }

    if "items" in schema:
        converted["items"] = convert_json_schema_to_google_schema(schema["items"])

    if "required" in schema:
        converted["required"] = list(schema["required"])

    if "enum" in schema:
        converted["enum"] = list(schema["enum"])

    if "description" in schema and "properties" not in schema:
        converted["description"] = schema["description"]

    return converted


def apply_google_batch_reasoning(generation_config: dict, reasoning_cfg) -> dict:
    """Apply Gemini batch-compatible thinking controls.

    File-based Gemini batch submission rejects `thinking_level` in uploaded
    JSONL requests, but accepts `thinking_budget`.
    """
    payload = deepcopy(generation_config)
    if reasoning_cfg is None:
        return payload
    if reasoning_cfg.effort is not None and reasoning_cfg.tokens is not None:
        raise ValueError(
            "Google Gemini requests cannot set both reasoning.effort and reasoning.tokens in the same model config."
        )
    thinking_config: dict[str, int] = {}
    if reasoning_cfg.tokens is not None:
        thinking_config["thinking_budget"] = reasoning_cfg.tokens
    elif reasoning_cfg.effort is not None:
        if reasoning_cfg.effort == "low":
            thinking_config["thinking_budget"] = 128
        elif reasoning_cfg.effort == "high":
            pass
        else:
            raise ValueError(
                "Google batch requests currently support reasoning.effort='low' only; "
                "use reasoning.tokens for other Google batch reasoning settings."
            )
    if thinking_config:
        payload["thinking_config"] = thinking_config
    return payload


def run_repeats(
    model: str,
    schema_name: str,
    repeats: int,
    temperature: float,
    max_tokens: int,
    chunk_size: int,
    max_retries: int,
    initial_backoff: float,
    messages_template: list[dict],
    seed: int | None = None,
    extra_request_kwargs: dict | None = None,
) -> list[dict]:
    """Run repeated async completions via LiteLLM batch_completion."""
    get_schema(schema_name)
    results: dict[int, dict] = {}
    pending_indices = list(range(repeats))
    retry_count = 0
    backoff = initial_backoff

    while pending_indices and retry_count <= max_retries:
        if retry_count > 0:
            logger.warning(
                f"Retry {retry_count}/{max_retries}: {len(pending_indices)} pending requests, backing off {backoff:.1f}s"
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, 120.0)

        still_pending: list[int] = []
        for chunk_start in range(0, len(pending_indices), chunk_size):
            chunk_indices = pending_indices[chunk_start : chunk_start + chunk_size]
            messages = [deepcopy(messages_template) for _ in chunk_indices]
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if seed is not None:
                kwargs["seed"] = seed
            if extra_request_kwargs:
                kwargs.update(extra_request_kwargs)

            try:
                responses = batch_completion(**kwargs)
            except Exception as exc:
                logger.warning(f"Chunk failed with exception: {exc}")
                still_pending.extend(chunk_indices)
                continue

            for idx, response in zip(chunk_indices, responses):
                if isinstance(response, Exception):
                    err_str = str(response).lower()
                    if "rate" in err_str or "429" in err_str or "limit" in err_str:
                        still_pending.append(idx)
                        continue
                    results[idx] = {
                        "run_index": idx,
                        "raw": f"ERROR: {response}",
                        "json": make_refusal_response(),
                    }
                    continue

                content = response.choices[0].message.content
                if isinstance(content, list):
                    raw = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                else:
                    raw = str(content) if content is not None else ""
                extracted = extract_json(raw)
                results[idx] = {
                    "run_index": idx,
                    "raw": raw,
                    "json": extracted if extracted else make_refusal_response(),
                }

            if chunk_start + chunk_size < len(pending_indices):
                time.sleep(0.5)

        pending_indices = still_pending
        if still_pending:
            retry_count += 1

    for idx in pending_indices:
        results[idx] = {
            "run_index": idx,
            "raw": "ERROR: Max retries exceeded",
            "json": make_refusal_response(),
        }

    return [results[i] for i in range(repeats)]


def create_openai_batch(
    experiment_name: str,
    model_name: str,
    tasks: list[ExecutionTask],
    repeats: int,
    temperature: float,
    output_tokens: int,
    use_structured_output: bool,
    batch_input_path: Path,
    seed: int | None = None,
    reasoning=None,
) -> dict:
    """Create and submit an OpenAI batch job."""
    requests: list[dict] = []
    request_id_map: dict[str, dict[str, int | str]] = {}
    for repeat in range(1, repeats + 1):
        for task_index, task in enumerate(tasks, start=1):
            schema_cls = get_schema(task.schema_name)
            request_id = build_batch_request_id(task_index, repeat)
            body = {
                "model": model_name,
                "messages": build_openai_messages(task.system_prompt, task.user_message),
            }
            body = apply_openai_reasoning(body, output_tokens, reasoning)
            if model_name not in OPENAI_NO_TEMP_MODELS:
                body["temperature"] = temperature
            if seed is not None:
                body["seed"] = seed
            if use_structured_output:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_cls.__name__,
                        "schema": schema_cls.model_json_schema(),
                        "strict": True,
                    },
                }
            requests.append(
                {
                    "custom_id": request_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )
            request_id_map[request_id] = {"request_stem": task.request_stem, "repeat": repeat}

    batch_input_path.write_text("".join(json.dumps(request) + "\n" for request in requests))
    client = OpenAI()
    with open(batch_input_path, "rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "experiment_name": experiment_name,
            "num_tasks": str(len(tasks)),
            "num_repeats": str(repeats),
        },
    )

    return {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "status": batch.status,
        "num_requests": len(requests),
        "request_id_map": request_id_map,
    }


def create_anthropic_batch(
    model_name: str,
    tasks: list[ExecutionTask],
    repeats: int,
    temperature: float,
    output_tokens: int,
    batch_input_path: Path,
    reasoning=None,
) -> dict:
    """Create and submit an Anthropic batch job."""
    requests: list[dict] = []
    request_id_map: dict[str, dict[str, int | str]] = {}
    for repeat in range(1, repeats + 1):
        for task_index, task in enumerate(tasks, start=1):
            request_id = build_batch_request_id(task_index, repeat)
            params = {
                "model": model_name,
                "temperature": temperature,
                "messages": build_anthropic_messages(task.user_message),
            }
            params = apply_anthropic_reasoning(params, output_tokens, reasoning)
            if task.system_prompt:
                params["system"] = task.system_prompt
            requests.append(
                {
                    "custom_id": request_id,
                    "params": params,
                }
            )
            request_id_map[request_id] = {"request_stem": task.request_stem, "repeat": repeat}

    batch_input_path.write_text("".join(json.dumps(request) + "\n" for request in requests))
    client = Anthropic()
    batch = client.messages.batches.create(requests=requests)
    return {
        "batch_id": batch.id,
        "status": batch.processing_status,
        "num_requests": len(requests),
        "request_id_map": request_id_map,
    }


def create_google_batch(
    model_name: str,
    tasks: list[ExecutionTask],
    repeats: int,
    temperature: float,
    output_tokens: int,
    use_structured_output: bool,
    batch_input_path: Path,
    reasoning=None,
) -> dict:
    """Create and submit a Gemini batch job."""
    from google.genai import types as gemini_types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable 'GEMINI_API_KEY' must be set for Gemini batching.")

    requests: list[dict] = []
    request_id_map: dict[str, dict[str, int | str]] = {}
    for repeat in range(1, repeats + 1):
        for task_index, task in enumerate(tasks, start=1):
            schema_cls = get_schema(task.schema_name)
            request_id = build_batch_request_id(task_index, repeat)
            generation_config = apply_google_batch_reasoning(
                {
                    "temperature": temperature,
                    "max_output_tokens": output_tokens,
                },
                reasoning,
            )
            if use_structured_output:
                generation_config["response_mime_type"] = "application/json"
                generation_config["response_schema"] = convert_json_schema_to_google_schema(
                    schema_cls.model_json_schema()
                )
            request = {
                "key": request_id,
                "request": {
                    "contents": build_google_contents(task.user_message),
                    "generation_config": generation_config,
                },
            }
            if task.system_prompt:
                request["request"]["system_instruction"] = {
                    "parts": [{"text": task.system_prompt}]
                }
            requests.append(request)
            request_id_map[request_id] = {"request_stem": task.request_stem, "repeat": repeat}

    batch_input_path.write_text("".join(json.dumps(request) + "\n" for request in requests))
    client = genai.Client(api_key=api_key)
    upload = client.files.upload(
        file=str(batch_input_path),
        config=gemini_types.UploadFileConfig(
            display_name="vignettes_batch_input",
            mime_type="jsonl",
        ),
    )
    batch = client.batches.create(model=model_name, src=upload.name)
    return {
        "batch_id": batch.name,
        "file_name": upload.name,
        "status": batch.state.name if hasattr(batch, "state") else "SUBMITTED",
        "num_requests": len(requests),
        "request_id_map": request_id_map,
    }
