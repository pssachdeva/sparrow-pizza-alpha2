"""Shared helpers for the law_norms_llms project."""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"
DATASETS_DIR = PROJECT_ROOT / "datasets"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def datasets_dir_for_run_dir(run_dir: Path) -> Path:
    """Infer the datasets directory that should accompany one run directory."""
    if run_dir.parent.name == "runs":
        return run_dir.parent.parent / "datasets"
    return run_dir.parent / "datasets"


def expand_path_pattern(path_pattern: str | Path) -> list[Path]:
    """Expand a literal path or glob pattern into sorted matches."""
    pattern = resolve_path(path_pattern)
    matches = sorted(Path(match) for match in glob.glob(str(pattern)))
    if matches:
        return matches
    if pattern.exists():
        return [pattern]
    raise FileNotFoundError(f"No files matched path pattern: {path_pattern}")


def slugify(text: str) -> str:
    """Convert text into a compact filename-safe identifier."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "item"


def value_slug(value: Any) -> str:
    """Convert a case value into a filename-safe token."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value}".replace("-", "m").replace(".", "p")
    return slugify(str(value))


def build_case_slug(case_values: dict[str, Any]) -> str:
    """Build a stable identifier for one concrete threshold assignment."""
    if not case_values:
        return "base"
    parts = [f"{slugify(key)}_{value_slug(value)}" for key, value in case_values.items()]
    return "__".join(parts)


def build_request_stem(prompt_name: str, vignette_name: str, case_slug: str) -> str:
    """Build a stable request identifier stem."""
    return (
        f"{slugify(prompt_name)}__vignette_{slugify(vignette_name)}__case_{case_slug}"
    )


def sanitize_model_name(model_name: str) -> str:
    """Convert provider model names into safe directory/file identifiers."""
    return model_name.replace("/", "_").replace(":", "_")


def format_temperature_suffix(temperature: float) -> str:
    """Format a temperature value into a stable suffix."""
    text = f"{temperature:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def write_json(path: Path, payload: Any) -> None:
    """Write indented JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_text(path: Path) -> str:
    """Read UTF-8 text from disk."""
    return path.read_text()
