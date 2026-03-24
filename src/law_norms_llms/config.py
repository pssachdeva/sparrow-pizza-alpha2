"""Configuration models for vignette experiments."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from law_norms_llms.utils import PROJECT_ROOT, RUNS_DIR


ThresholdScalar = int | float | str


class PromptConfig(BaseModel):
    """A single prompt specification for vignette experiments."""

    model_config = ConfigDict(extra="forbid")

    name: str
    schema_name: str
    system_template_path: Path | None = None


class VignetteSourceConfig(BaseModel):
    """A vignette source specification backed by a literal file or glob."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    path: str
    thresholds: dict[str, list[ThresholdScalar]] = Field(default_factory=dict)


class ReasoningConfig(BaseModel):
    """Provider-level reasoning controls."""

    model_config = ConfigDict(extra="forbid")

    effort: str | None = None
    tokens: int | None = Field(default=None, ge=0)


class ModelConfig(BaseModel):
    """A single model target."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    name: str
    temperature: float | None = Field(default=None, ge=0, le=2)
    output_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    seed: int | None = None
    use_structured_output: bool | None = None
    reasoning: ReasoningConfig | None = None


class VignetteExperimentConfig(BaseModel):
    """Top-level config for a self-contained vignette experiment."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    description: str | None = None
    run_dir: Path | None = None
    prompts: list[PromptConfig]
    vignettes: list[VignetteSourceConfig]
    models: list[ModelConfig]
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_completion_tokens: int = Field(default=500, ge=1)
    use_structured_output: bool = True
    seed: int | None = None
    repeats: int = Field(default=1, ge=1)
    concurrency: int = Field(default=5, ge=1)
    max_retries: int = Field(default=10, ge=0)
    initial_backoff: float = Field(default=5.0, ge=0.1)

    @model_validator(mode="after")
    def validate_unique_prompt_names(self) -> "VignetteExperimentConfig":
        """Ensure prompt names are unique."""
        names = [prompt.name for prompt in self.prompts]
        if len(names) != len(set(names)):
            raise ValueError("Prompt names must be unique.")
        return self

    def resolved_run_dir(self) -> Path:
        """Return the concrete run directory for this experiment."""
        if self.run_dir is None:
            return RUNS_DIR / self.experiment_name
        if self.run_dir.is_absolute():
            return self.run_dir
        return PROJECT_ROOT / self.run_dir


def load_experiment_config(config_path: str | Path) -> VignetteExperimentConfig:
    """Load a vignette experiment YAML file."""
    payload = yaml.safe_load(Path(config_path).read_text())
    return VignetteExperimentConfig(**payload)
