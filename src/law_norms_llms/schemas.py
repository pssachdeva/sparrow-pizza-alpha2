"""Pydantic schemas for vignette experiment outputs."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Strict base model suitable for structured-output providers."""

    model_config = ConfigDict(extra="forbid")


class VignetteResponse(StrictModel):
    """Single-item appropriateness response."""

    answer: Literal["A", "B", "C", "D"]


SCHEMA_REGISTRY = {
    "VignetteResponse": VignetteResponse,
    "ExampleResponse": VignetteResponse,
}


def get_schema(name: str):
    """Look up a local schema by name."""
    if name not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown schema: {name}. Available: {list(SCHEMA_REGISTRY.keys())}")
    return SCHEMA_REGISTRY[name]


def make_refusal_response() -> str:
    """Return a sentinel response when the provider does not return valid JSON."""
    return json.dumps({"answer": "REFUSAL"})
