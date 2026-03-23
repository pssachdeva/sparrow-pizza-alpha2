"""Provider-agnostic helpers for vignette runners."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from law_norms_llms.utils import format_temperature_suffix


OPENAI_NO_TEMP_MODELS = {"gpt-5-2025-08-07"}


def extract_json(text: str) -> str | None:
    """Extract the first complete JSON object from a string."""
    if "```" in text:
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.startswith("```"))

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for index, char in enumerate(text[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def model_temperature(model_cfg, default_temp: float) -> float:
    """Resolve the effective temperature for a model."""
    return model_cfg.temperature if model_cfg.temperature is not None else default_temp


def model_max_tokens(model_cfg, default_tokens: int) -> int:
    """Resolve the legacy combined completion token setting for a model."""
    return (
        model_cfg.max_completion_tokens
        if model_cfg.max_completion_tokens is not None
        else default_tokens
    )


def model_output_tokens(model_cfg, default_tokens: int) -> int:
    """Resolve the visible output token budget for a model."""
    if getattr(model_cfg, "output_tokens", None) is not None:
        return model_cfg.output_tokens
    return model_max_tokens(model_cfg, default_tokens)


def model_seed(model_cfg, default_seed: int | None) -> int | None:
    """Resolve the effective sampling seed for a model."""
    return model_cfg.seed if model_cfg.seed is not None else default_seed


def model_use_structured_output(model_cfg, default_value: bool) -> bool:
    """Resolve whether a model should request structured output."""
    if model_cfg.use_structured_output is None:
        return default_value
    return model_cfg.use_structured_output


def model_id_for(model_cfg, default_temp: float) -> str:
    """Build a stable model identifier that includes per-model temperature overrides."""
    if model_cfg.temperature is None:
        return model_cfg.name
    suffix = format_temperature_suffix(model_temperature(model_cfg, default_temp))
    return f"{model_cfg.name}__t{suffix}"


def build_openai_messages(system_prompt: str | None, user_message: str) -> list[dict[str, Any]]:
    """Build OpenAI-style messages."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    return messages


def build_anthropic_messages(user_message: str) -> list[dict[str, Any]]:
    """Build Anthropic-style message content."""
    return [{"role": "user", "content": [{"type": "text", "text": user_message}]}]


def build_google_contents(user_message: str) -> list[dict[str, Any]]:
    """Build Gemini-style contents list."""
    return [{"parts": [{"text": user_message}], "role": "user"}]


def model_reasoning(model_cfg):
    """Return the model-level reasoning config, if present."""
    return getattr(model_cfg, "reasoning", None)


def reasoning_metadata(reasoning_cfg) -> dict[str, Any] | None:
    """Convert reasoning config into JSON-serializable metadata."""
    if reasoning_cfg is None:
        return None
    payload = {}
    if reasoning_cfg.effort is not None:
        payload["effort"] = reasoning_cfg.effort
    if reasoning_cfg.tokens is not None:
        payload["tokens"] = reasoning_cfg.tokens
    return payload or None


def openai_max_completion_tokens(output_tokens: int, reasoning_cfg) -> int:
    """Approximate total OpenAI completion budget from output and reasoning headroom."""
    if reasoning_cfg and reasoning_cfg.tokens is not None:
        return output_tokens + reasoning_cfg.tokens
    return output_tokens


def anthropic_max_tokens(output_tokens: int, reasoning_cfg) -> int:
    """Compute Anthropic max_tokens from output and thinking budget."""
    if reasoning_cfg and reasoning_cfg.tokens is not None:
        return output_tokens + reasoning_cfg.tokens
    return output_tokens


def provider_token_budget(provider: str, output_tokens: int, reasoning_cfg) -> int:
    """Return the provider-level token cap that will be sent on the request."""
    if provider == "openai":
        return openai_max_completion_tokens(output_tokens, reasoning_cfg)
    if provider == "anthropic":
        return anthropic_max_tokens(output_tokens, reasoning_cfg)
    if provider == "google":
        return output_tokens
    return output_tokens


def apply_openai_reasoning(body: dict[str, Any], output_tokens: int, reasoning_cfg) -> dict[str, Any]:
    """Apply OpenAI-specific reasoning settings to a request body."""
    payload = deepcopy(body)
    payload["max_completion_tokens"] = openai_max_completion_tokens(output_tokens, reasoning_cfg)
    if reasoning_cfg and reasoning_cfg.effort is not None:
        payload["reasoning_effort"] = reasoning_cfg.effort
    return payload


def apply_anthropic_reasoning(params: dict[str, Any], output_tokens: int, reasoning_cfg) -> dict[str, Any]:
    """Apply Anthropic-specific thinking settings to request params."""
    payload = deepcopy(params)
    payload["max_tokens"] = anthropic_max_tokens(output_tokens, reasoning_cfg)
    if reasoning_cfg is None:
        return payload
    if reasoning_cfg.effort is not None:
        raise ValueError("Anthropic reasoning.effort is not supported; use reasoning.tokens instead.")
    if reasoning_cfg.tokens is not None:
        payload["thinking"] = {
            "type": "enabled",
            "budget_tokens": reasoning_cfg.tokens,
        }
    return payload


def apply_google_reasoning(
    generation_config: dict[str, Any],
    output_tokens: int,
    reasoning_cfg,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Apply Gemini-specific thinking settings to generation config."""
    payload = deepcopy(generation_config)
    payload["max_output_tokens"] = output_tokens
    if reasoning_cfg is None:
        return payload
    if reasoning_cfg.effort is not None and reasoning_cfg.tokens is not None:
        raise ValueError(
            "Google Gemini requests cannot set both reasoning.effort and reasoning.tokens in the same model config."
        )
    thinking_config: dict[str, Any] = {}
    if reasoning_cfg.effort is not None:
        thinking_config["thinking_level"] = reasoning_cfg.effort
    if reasoning_cfg.tokens is not None:
        thinking_config["thinking_budget"] = reasoning_cfg.tokens
    if thinking_config:
        payload["thinking_config"] = thinking_config
    return payload


def async_reasoning_kwargs(provider: str, output_tokens: int, reasoning_cfg) -> dict[str, Any]:
    """Build provider-specific extra kwargs for async/LiteLLM execution."""
    if reasoning_cfg is None:
        return {}
    if provider == "openai":
        payload: dict[str, Any] = {
            "max_completion_tokens": openai_max_completion_tokens(output_tokens, reasoning_cfg),
        }
        if reasoning_cfg.effort is not None:
            payload["reasoning_effort"] = reasoning_cfg.effort
        return payload
    if provider == "anthropic":
        if reasoning_cfg.effort is not None:
            raise ValueError("Anthropic reasoning.effort is not supported; use reasoning.tokens instead.")
        payload = {
            "max_tokens": anthropic_max_tokens(output_tokens, reasoning_cfg),
        }
        if reasoning_cfg.tokens is not None:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": reasoning_cfg.tokens,
            }
        return payload
    if provider == "google":
        return {
            "max_output_tokens": output_tokens,
            "thinking_config": apply_google_reasoning({}, output_tokens, reasoning_cfg).get("thinking_config"),
        }
    return {}
