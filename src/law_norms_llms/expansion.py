"""Expand config-defined vignette groups into concrete execution tasks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

from law_norms_llms.config import VignetteExperimentConfig
from law_norms_llms.templating import missing_placeholders, render_template
from law_norms_llms.utils import (
    build_case_slug,
    build_request_stem,
    expand_path_pattern,
    load_text,
    resolve_path,
)


@dataclass(frozen=True)
class ConcreteVignette:
    """A single concrete vignette file with shared thresholds."""

    vignette_name: str
    vignette_path: Path
    thresholds: dict[str, list[Any]]


@dataclass(frozen=True)
class ExecutionTask:
    """A single prompt × vignette × threshold assignment to execute."""

    prompt_name: str
    schema_name: str
    system_prompt: str | None
    user_message: str
    vignette_name: str
    vignette_path: str
    case_slug: str
    case_values: dict[str, Any]
    request_stem: str

    def manifest_entry(self) -> dict[str, Any]:
        """Convert a task into a JSON-serializable manifest payload."""
        return asdict(self)


def expand_vignettes(cfg: VignetteExperimentConfig) -> list[ConcreteVignette]:
    """Expand literal and glob-backed vignette definitions into concrete files."""
    concrete: list[ConcreteVignette] = []
    for source in cfg.vignettes:
        matches = expand_path_pattern(source.path)
        if source.name and len(matches) > 1:
            raise ValueError(f"Vignette source '{source.path}' matched multiple files and cannot use one explicit name.")
        for match in matches:
            concrete.append(
                ConcreteVignette(
                    vignette_name=source.name or match.stem,
                    vignette_path=match,
                    thresholds=source.thresholds,
                )
            )
    return concrete


def expand_case_values(thresholds: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a vignette's threshold grid into concrete assignments."""
    if not thresholds:
        return [{}]

    items = list(thresholds.items())
    keys = [key for key, _ in items]
    value_lists = [values for _, values in items]
    return [dict(zip(keys, combination, strict=True)) for combination in product(*value_lists)]


def build_execution_tasks(cfg: VignetteExperimentConfig) -> list[ExecutionTask]:
    """Expand and validate all prompt/vignette combinations before any execution."""
    tasks: list[ExecutionTask] = []
    seen_request_stems: set[str] = set()

    for prompt in cfg.prompts:
        system_template = (
            load_text(resolve_path(prompt.system_template_path))
            if prompt.system_template_path
            else None
        )

        for vignette in expand_vignettes(cfg):
            raw_vignette = load_text(vignette.vignette_path)

            for case_values in expand_case_values(vignette.thresholds):
                case_slug = build_case_slug(case_values)
                vignette_missing = missing_placeholders(raw_vignette, set(case_values))
                if vignette_missing:
                    missing = ", ".join(sorted(vignette_missing))
                    raise ValueError(
                        f"Vignette '{vignette.vignette_path}' requires missing placeholders: {missing}"
                    )

                user_message = render_template(raw_vignette, case_values)
                system_prompt = None
                if system_template is not None:
                    system_context = {
                        "vignette": user_message,
                        "vignette_name": vignette.vignette_name,
                        "vignette_path": str(vignette.vignette_path),
                        "case_slug": case_slug,
                        **case_values,
                    }
                    system_missing = missing_placeholders(system_template, set(system_context))
                    if system_missing:
                        missing = ", ".join(sorted(system_missing))
                        raise ValueError(
                            f"System template '{prompt.system_template_path}' requires missing placeholders: {missing}"
                        )
                    system_prompt = render_template(system_template, system_context)

                request_stem = build_request_stem(prompt.name, vignette.vignette_name, case_slug)
                if request_stem in seen_request_stems:
                    raise ValueError(f"Duplicate request stem generated: {request_stem}")
                seen_request_stems.add(request_stem)

                tasks.append(
                    ExecutionTask(
                        prompt_name=prompt.name,
                        schema_name=prompt.schema_name,
                        system_prompt=system_prompt,
                        user_message=user_message,
                        vignette_name=vignette.vignette_name,
                        vignette_path=str(vignette.vignette_path),
                        case_slug=case_slug,
                        case_values=case_values,
                        request_stem=request_stem,
                    )
                )

    return tasks
