"""Template parsing and rendering helpers for vignette experiments."""
from __future__ import annotations

from string import Formatter


def extract_placeholders(text: str) -> set[str]:
    """Return all replacement field names used in a format-style template."""
    placeholders: set[str] = set()
    formatter = Formatter()
    for _, field_name, _, _ in formatter.parse(text):
        if field_name:
            placeholders.add(field_name)
    return placeholders


def missing_placeholders(text: str, available_keys: set[str]) -> set[str]:
    """Return placeholders referenced by text but absent from the context."""
    return extract_placeholders(text) - available_keys


def render_template(text: str, values: dict[str, object]) -> str:
    """Render a format-style template with strict missing-key behavior."""
    return text.format(**values)
