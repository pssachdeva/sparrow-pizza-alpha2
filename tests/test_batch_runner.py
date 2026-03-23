from pathlib import Path
import json

import pytest
import yaml

from law_norms_llms import batch_runner
from law_norms_llms.batch import apply_google_batch_reasoning, create_google_batch
from law_norms_llms.expansion import ExecutionTask
from law_norms_llms.providers import (
    apply_anthropic_reasoning,
    apply_google_reasoning,
    apply_openai_reasoning,
)


def write_config(tmp_path: Path, payload: dict) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload))
    return config_path


def test_batch_runner_fails_before_writing_batch_input_on_preflight_error(tmp_path: Path):
    system_path = tmp_path / "system.txt"
    system_path.write_text("Return JSON.")
    vignette_path = tmp_path / "case.txt"
    vignette_path.write_text("Customer is {age}.")
    run_dir = tmp_path / "runs"

    config_path = write_config(
        tmp_path,
        {
            "experiment_name": "demo",
            "run_dir": str(run_dir),
            "prompts": [
                {
                    "name": "appropriateness",
                    "schema_name": "VignetteResponse",
                    "system_template_path": str(system_path),
                }
            ],
            "vignettes": [
                {
                    "path": str(vignette_path),
                    "thresholds": {"weight": [7.5]},
                }
            ],
            "models": [{"provider": "openai", "name": "gpt-5.1"}],
        },
    )

    with pytest.raises(ValueError):
        batch_runner.main(str(config_path))

    assert not run_dir.exists()


def test_batch_runner_writes_manifest_and_metadata(tmp_path: Path, monkeypatch):
    system_path = tmp_path / "system.txt"
    system_path.write_text("Return JSON.")
    vignette_path = tmp_path / "case.txt"
    vignette_path.write_text("Customer is {age}.")
    run_dir = tmp_path / "runs"

    config_path = write_config(
        tmp_path,
        {
            "experiment_name": "demo",
            "run_dir": str(run_dir),
            "repeats": 2,
            "prompts": [
                {
                    "name": "appropriateness",
                    "schema_name": "VignetteResponse",
                    "system_template_path": str(system_path),
                }
            ],
            "vignettes": [
                {
                    "path": str(vignette_path),
                    "thresholds": {"age": [17, 18]},
                }
            ],
            "models": [{"provider": "openai", "name": "gpt-5.1"}],
        },
    )

    def fake_create_openai_batch(**kwargs):
        return {
            "batch_id": "batch_123",
            "input_file_id": "file_123",
            "status": "submitted",
            "num_requests": len(kwargs["tasks"]) * kwargs["repeats"],
            "request_id_map": {
                "req00001_rep001": {"request_stem": "appropriateness__vignette_case__case_age_17", "repeat": 1}
            },
        }

    monkeypatch.setattr(batch_runner, "create_openai_batch", fake_create_openai_batch)
    batch_runner.main(str(config_path))

    assert (run_dir / "execution_manifest.json").exists()
    assert (run_dir / "batch_metadata.json").exists()
    request_maps = list(run_dir.glob("request_id_map_*.json"))
    assert len(request_maps) == 1


def test_batch_runner_persists_successful_batches_before_later_failure(tmp_path: Path, monkeypatch):
    system_path = tmp_path / "system.txt"
    system_path.write_text("Return JSON.")
    vignette_path = tmp_path / "case.txt"
    vignette_path.write_text("Customer is {age}.")
    run_dir = tmp_path / "runs"

    config_path = write_config(
        tmp_path,
        {
            "experiment_name": "demo",
            "run_dir": str(run_dir),
            "prompts": [
                {
                    "name": "appropriateness",
                    "schema_name": "VignetteResponse",
                    "system_template_path": str(system_path),
                }
            ],
            "vignettes": [
                {
                    "path": str(vignette_path),
                    "thresholds": {"age": [17]},
                }
            ],
            "models": [
                {"provider": "openai", "name": "gpt-5.1"},
                {"provider": "google", "name": "gemini-3-flash-preview"},
            ],
        },
    )

    monkeypatch.setattr(
        batch_runner,
        "create_openai_batch",
        lambda **kwargs: {
            "batch_id": "batch_123",
            "input_file_id": "file_123",
            "status": "submitted",
            "num_requests": len(kwargs["tasks"]) * kwargs["repeats"],
            "request_id_map": {
                "req00001_rep001": {
                    "request_stem": "appropriateness__vignette_case__case_age_17",
                    "repeat": 1,
                }
            },
        },
    )

    def fail_google_batch(**kwargs):
        raise RuntimeError("google submission failed")

    monkeypatch.setattr(batch_runner, "create_google_batch", fail_google_batch)

    with pytest.raises(RuntimeError, match="google submission failed"):
        batch_runner.main(str(config_path))

    metadata = json.loads((run_dir / "batch_metadata.json").read_text())
    assert len(metadata["batches"]) == 1
    assert metadata["batches"][0]["provider"] == "openai"


def test_create_google_batch_writes_json_serializable_schema(tmp_path: Path, monkeypatch):
    batch_input_path = tmp_path / "batch_input.jsonl"
    upload_configs = []
    task = ExecutionTask(
        prompt_name="appropriateness",
        schema_name="VignetteResponse",
        system_prompt="Return JSON.",
        user_message="Customer is 17.",
        vignette_name="casino_bad",
        vignette_path=str(tmp_path / "casino_bad.txt"),
        case_slug="age_17",
        case_values={"age": 17},
        request_stem="appropriateness__vignette_casino_bad__case_age_17",
    )

    class DummyUpload:
        name = "files/upload-123"

    class DummyState:
        name = "SUBMITTED"

    class DummyBatch:
        name = "batches/123"
        state = DummyState()

    class DummyFiles:
        def upload(self, file, config):
            upload_configs.append(config)
            return DummyUpload()

    class DummyBatches:
        def create(self, model, src):
            return DummyBatch()

    class DummyClient:
        def __init__(self, api_key):
            self.files = DummyFiles()
            self.batches = DummyBatches()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("law_norms_llms.batch.genai.Client", DummyClient)

    result = create_google_batch(
        model_name="gemini-3-flash-preview",
        tasks=[task],
        repeats=1,
        temperature=1.0,
        output_tokens=100,
        use_structured_output=True,
        batch_input_path=batch_input_path,
    )

    assert result["num_requests"] == 1
    payload = yaml.safe_load(batch_input_path.read_text().strip())
    assert payload["request"]["generation_config"]["response_schema"]["properties"]["answer"]["enum"] == [
        "A",
        "B",
        "C",
        "D",
    ]
    assert payload["request"]["generation_config"]["response_schema"]["type"] == "OBJECT"
    assert "additionalProperties" not in payload["request"]["generation_config"]["response_schema"]
    assert payload["request"]["system_instruction"]["parts"][0]["text"] == "Return JSON."
    assert upload_configs[0].mime_type == "jsonl"


def test_reasoning_mapping_is_provider_specific():
    class Reasoning:
        def __init__(self, effort=None, tokens=None):
            self.effort = effort
            self.tokens = tokens

    openai_payload = apply_openai_reasoning(
        {"model": "gpt-5.1"},
        output_tokens=100,
        reasoning_cfg=Reasoning(effort="medium", tokens=200),
    )
    assert openai_payload["max_completion_tokens"] == 300
    assert openai_payload["reasoning_effort"] == "medium"

    anthropic_payload = apply_anthropic_reasoning(
        {"model": "claude-sonnet"},
        output_tokens=120,
        reasoning_cfg=Reasoning(tokens=400),
    )
    assert anthropic_payload["max_tokens"] == 520
    assert anthropic_payload["thinking"]["budget_tokens"] == 400

    google_payload = apply_google_reasoning(
        {"temperature": 1.0},
        output_tokens=80,
        reasoning_cfg=Reasoning(effort="minimal"),
    )
    assert google_payload["max_output_tokens"] == 80
    assert google_payload["thinking_config"]["thinking_level"] == "minimal"


def test_google_reasoning_rejects_effort_and_tokens_together():
    class Reasoning:
        def __init__(self, effort=None, tokens=None):
            self.effort = effort
            self.tokens = tokens

    with pytest.raises(ValueError, match="cannot set both reasoning.effort and reasoning.tokens"):
        apply_google_reasoning(
            {"temperature": 1.0},
            output_tokens=80,
            reasoning_cfg=Reasoning(effort="minimal", tokens=256),
        )


def test_google_batch_reasoning_maps_low_effort_to_budget():
    class Reasoning:
        def __init__(self, effort=None, tokens=None):
            self.effort = effort
            self.tokens = tokens

    payload = apply_google_batch_reasoning(
        {"temperature": 1.0, "max_output_tokens": 80},
        Reasoning(effort="low"),
    )
    assert payload["thinking_config"]["thinking_budget"] == 128
