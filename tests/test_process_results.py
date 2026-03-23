import json
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest
import yaml

import law_norms_llms.process_results as process_results
from law_norms_llms.process_results import (
    check_google_batch_status,
    create_results_dataframe,
    main,
    parse_and_save_batch_results,
    process_batch_outputs,
)
from law_norms_llms.utils import write_json


def write_config(tmp_path: Path, payload: dict) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload))
    return config_path


def test_parse_and_save_batch_results_materializes_per_run_json(tmp_path: Path):
    experiment_dir = tmp_path / "results"
    experiment_dir.mkdir()
    output_path = experiment_dir / "batch_output.jsonl"
    output_path.write_text(
        json.dumps(
            {
                "custom_id": "req00001_rep001",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {"message": {"content": '{"answer":"A"}'}}
                        ]
                    },
                },
            }
        )
        + "\n"
    )

    manifest = {
        "appropriateness__vignette_casino_bad__case_age_17": {
            "prompt_name": "appropriateness",
            "schema_name": "VignetteResponse",
            "vignette_name": "casino_bad",
            "vignette_path": "/tmp/casino_bad.txt",
            "case_slug": "age_17",
            "case_values": {"age": 17},
        }
    }
    batch_entry = {
        "provider": "openai",
        "model": "gpt-5.1",
        "model_id": "gpt-5.1",
        "temperature": 0.0,
        "max_completion_tokens": 1000,
        "request_id_map_path": str(experiment_dir / "request_id_map_openai_gpt-5.1.json"),
    }
    write_json(
        experiment_dir / "request_id_map_openai_gpt-5.1.json",
        {
            "req00001_rep001": {
                "request_stem": "appropriateness__vignette_casino_bad__case_age_17",
                "repeat": 1,
            }
        },
    )

    parse_and_save_batch_results(output_path, experiment_dir, batch_entry, manifest)
    result_files = list((experiment_dir / "openai_gpt-5.1").glob("*_repeat_*.json"))
    assert len(result_files) == 1


def test_parse_and_save_batch_results_skips_errored_anthropic_rows(tmp_path: Path):
    experiment_dir = tmp_path / "results"
    experiment_dir.mkdir()
    output_path = experiment_dir / "batch_output.jsonl"
    output_path.write_text(
        json.dumps(
            {
                "custom_id": "req00001_rep001",
                "result": {
                    "type": "errored",
                    "error": {
                        "error": {
                            "message": "thinking.enabled.budget_tokens: Input should be greater than or equal to 1024",
                        }
                    },
                },
            }
        )
        + "\n"
    )

    manifest = {
        "appropriateness__vignette_casino_bad__case_age_17": {
            "prompt_name": "appropriateness",
            "schema_name": "VignetteResponse",
            "vignette_name": "casino_bad",
            "vignette_path": "/tmp/casino_bad.txt",
            "case_slug": "age_17",
            "case_values": {"age": 17},
        }
    }
    batch_entry = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "model_id": "claude-sonnet-4-6",
        "temperature": 1.0,
        "max_completion_tokens": 3000,
        "request_id_map_path": str(experiment_dir / "request_id_map_anthropic_claude-sonnet-4-6.json"),
    }
    write_json(
        experiment_dir / "request_id_map_anthropic_claude-sonnet-4-6.json",
        {
            "req00001_rep001": {
                "request_stem": "appropriateness__vignette_casino_bad__case_age_17",
                "repeat": 1,
            }
        },
    )

    parse_and_save_batch_results(output_path, experiment_dir, batch_entry, manifest)
    result_files = list((experiment_dir / "anthropic_claude-sonnet-4-6").glob("*_repeat_*.json"))
    assert result_files == []


def test_parse_and_save_batch_results_reads_anthropic_text_block_after_thinking(tmp_path: Path):
    experiment_dir = tmp_path / "results"
    experiment_dir.mkdir()
    output_path = experiment_dir / "batch_output.jsonl"
    output_path.write_text(
        json.dumps(
            {
                "custom_id": "req00001_rep001",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "content": [
                            {"type": "thinking", "thinking": "hidden reasoning"},
                            {"type": "text", "text": '```json\n{"answer":"D"}\n```'},
                        ]
                    },
                },
            }
        )
        + "\n"
    )

    manifest = {
        "appropriateness__vignette_casino_bad__case_age_17": {
            "prompt_name": "appropriateness",
            "schema_name": "VignetteResponse",
            "vignette_name": "casino_bad",
            "vignette_path": "/tmp/casino_bad.txt",
            "case_slug": "age_17",
            "case_values": {"age": 17},
        }
    }
    batch_entry = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "model_id": "claude-sonnet-4-6",
        "temperature": 1.0,
        "max_completion_tokens": 3000,
        "request_id_map_path": str(experiment_dir / "request_id_map_anthropic_claude-sonnet-4-6.json"),
    }
    write_json(
        experiment_dir / "request_id_map_anthropic_claude-sonnet-4-6.json",
        {
            "req00001_rep001": {
                "request_stem": "appropriateness__vignette_casino_bad__case_age_17",
                "repeat": 1,
            }
        },
    )

    parse_and_save_batch_results(output_path, experiment_dir, batch_entry, manifest)
    result_files = list((experiment_dir / "anthropic_claude-sonnet-4-6").glob("*_repeat_*.json"))
    assert len(result_files) == 1
    saved = json.loads(result_files[0].read_text())
    assert json.loads(saved["json"])["answer"] == "D"


def test_parse_and_save_batch_results_reads_google_output_by_line_order(tmp_path: Path):
    experiment_dir = tmp_path / "results"
    experiment_dir.mkdir()
    output_path = experiment_dir / "batch_output.jsonl"
    output_path.write_text(
        json.dumps(
            {
                "key": "req00001_rep001",
                "response": {"candidates": [{"content": {"parts": [{"text": '{"answer":"C"}'}]}}]},
            }
        )
        + "\n"
    )

    manifest = {
        "appropriateness__vignette_casino_bad__case_age_17": {
            "prompt_name": "appropriateness",
            "schema_name": "VignetteResponse",
            "vignette_name": "casino_bad",
            "vignette_path": "/tmp/casino_bad.txt",
            "case_slug": "age_17",
            "case_values": {"age": 17},
        }
    }
    batch_entry = {
        "provider": "google",
        "model": "gemini-3.1-pro-preview",
        "model_id": "gemini-3.1-pro-preview",
        "temperature": 1.0,
        "max_completion_tokens": 5000,
        "request_id_map_path": str(experiment_dir / "request_id_map_google_gemini-3.1-pro-preview.json"),
    }
    write_json(
        experiment_dir / "request_id_map_google_gemini-3.1-pro-preview.json",
        {
            "req00001_rep001": {
                "request_stem": "appropriateness__vignette_casino_bad__case_age_17",
                "repeat": 1,
            }
        },
    )

    parse_and_save_batch_results(output_path, experiment_dir, batch_entry, manifest)
    result_files = list((experiment_dir / "google_gemini-3.1-pro-preview").glob("*_repeat_*.json"))
    assert len(result_files) == 1
    saved = json.loads(result_files[0].read_text())
    assert json.loads(saved["json"])["answer"] == "C"


def test_process_results_creates_csv_with_dynamic_threshold_columns(tmp_path: Path):
    run_dir = tmp_path / "runs"
    datasets_dir = tmp_path / "datasets"
    model_dir = run_dir / "openai_gpt-5.1"
    model_dir.mkdir(parents=True)
    write_json(
        model_dir / "appropriateness__vignette_casino_bad__case_age_17_repeat_001.json",
        {
            "provider": "openai",
            "model": "gpt-5.1",
            "temperature": 0.0,
            "prompt_name": "appropriateness",
            "vignette_name": "casino_bad",
            "vignette_path": "/tmp/casino_bad.txt",
            "case_slug": "age_17",
            "case_values": {"age": 17},
            "repeat": 1,
            "raw": '{"answer":"A"}',
            "json": {"answer": "A"},
        },
    )
    write_json(
        model_dir / "appropriateness__vignette_lorry_bad__case_weight_7p5_repeat_001.json",
        {
            "provider": "openai",
            "model": "gpt-5.1",
            "temperature": 0.0,
            "prompt_name": "appropriateness",
            "vignette_name": "lorry_bad",
            "vignette_path": "/tmp/lorry_bad.txt",
            "case_slug": "weight_7p5",
            "case_values": {"weight": 7.5},
            "repeat": 1,
            "raw": '{"answer":"B"}',
            "json": {"answer": "B"},
        },
    )

    config_path = write_config(
        tmp_path,
        {
            "experiment_name": "demo",
            "run_dir": str(run_dir),
            "prompts": [{"name": "appropriateness", "schema_name": "VignetteResponse"}],
            "vignettes": [{"path": str(tmp_path / "*.txt")}],
            "models": [{"provider": "openai", "name": "gpt-5.1"}],
        },
    )

    dataframe = create_results_dataframe(run_dir)
    assert "age" in dataframe.columns
    assert "weight" in dataframe.columns
    assert len(dataframe) == 2

    main(str(config_path))
    assert (datasets_dir / "demo.csv").exists()


def test_main_writes_custom_output_csv(tmp_path: Path):
    run_dir = tmp_path / "runs"
    datasets_dir = tmp_path / "datasets"
    model_dir = run_dir / "openai_gpt-5.1"
    model_dir.mkdir(parents=True)
    write_json(
        model_dir / "appropriateness__vignette_casino_bad__case_age_17_repeat_001.json",
        {
            "provider": "openai",
            "model": "gpt-5.1",
            "temperature": 0.0,
            "prompt_name": "appropriateness",
            "vignette_name": "casino_bad",
            "vignette_path": "/tmp/casino_bad.txt",
            "case_slug": "age_17",
            "case_values": {"age": 17},
            "repeat": 1,
            "raw": '{"answer":"A"}',
            "json": {"answer": "A"},
        },
    )

    config_path = write_config(
        tmp_path,
        {
            "experiment_name": "demo",
            "run_dir": str(run_dir),
            "prompts": [{"name": "appropriateness", "schema_name": "VignetteResponse"}],
            "vignettes": [{"path": str(tmp_path / "*.txt")}],
            "models": [{"provider": "openai", "name": "gpt-5.1"}],
        },
    )

    main(str(config_path), output_csv="combined.csv")
    assert (datasets_dir / "combined.csv").exists()
    assert not (datasets_dir / "demo.csv").exists()


def test_main_appends_to_existing_output_csv(tmp_path: Path):
    run_dir = tmp_path / "runs"
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    model_dir = run_dir / "openai_gpt-5.1"
    model_dir.mkdir(parents=True)
    write_json(
        model_dir / "appropriateness__vignette_casino_bad__case_age_17_repeat_001.json",
        {
            "provider": "openai",
            "model": "gpt-5.1",
            "temperature": 0.0,
            "prompt_name": "appropriateness",
            "vignette_name": "casino_bad",
            "vignette_path": "/tmp/casino_bad.txt",
            "case_slug": "age_17",
            "case_values": {"age": 17},
            "repeat": 1,
            "raw": '{"answer":"A"}',
            "json": {"answer": "A"},
        },
    )

    config_path = write_config(
        tmp_path,
        {
            "experiment_name": "demo",
            "run_dir": str(run_dir),
            "prompts": [{"name": "appropriateness", "schema_name": "VignetteResponse"}],
            "vignettes": [{"path": str(tmp_path / "*.txt")}],
            "models": [{"provider": "openai", "name": "gpt-5.1"}],
        },
    )

    target_csv = datasets_dir / "combined.csv"
    pd.DataFrame(
        [
            {
                "provider": "openai",
                "model": "gpt-5.1",
                "temperature": 0.0,
                "prompt": "baseline",
                "vignette": "seatbelt",
                "vignette_path": "/tmp/seatbelt.txt",
                "case_slug": "",
                "repeat": 1,
                "item": "answer",
                "response": "B",
                "raw_output": '{"answer":"B"}',
                "age": None,
            }
        ]
    ).to_csv(target_csv, index=False)

    main(str(config_path), output_csv=str(target_csv), append=True)

    combined = pd.read_csv(target_csv)
    assert len(combined) == 2
    assert set(combined["vignette"]) == {"seatbelt", "casino_bad"}


def test_main_defaults_to_append_for_custom_output_csv(tmp_path: Path):
    run_dir = tmp_path / "runs"
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    model_dir = run_dir / "openai_gpt-5.1"
    model_dir.mkdir(parents=True)
    write_json(
        model_dir / "appropriateness__vignette_casino_bad__case_age_17_repeat_001.json",
        {
            "provider": "openai",
            "model": "gpt-5.1",
            "temperature": 0.0,
            "prompt_name": "appropriateness",
            "vignette_name": "casino_bad",
            "vignette_path": "/tmp/casino_bad.txt",
            "case_slug": "age_17",
            "case_values": {"age": 17},
            "repeat": 1,
            "raw": '{"answer":"A"}',
            "json": {"answer": "A"},
        },
    )

    config_path = write_config(
        tmp_path,
        {
            "experiment_name": "demo",
            "run_dir": str(run_dir),
            "prompts": [{"name": "appropriateness", "schema_name": "VignetteResponse"}],
            "vignettes": [{"path": str(tmp_path / "*.txt")}],
            "models": [{"provider": "openai", "name": "gpt-5.1"}],
        },
    )

    target_csv = datasets_dir / "combined.csv"
    pd.DataFrame(
        [
            {
                "provider": "openai",
                "model": "gpt-5.1",
                "temperature": 0.0,
                "prompt": "baseline",
                "vignette": "seatbelt",
                "vignette_path": "/tmp/seatbelt.txt",
                "case_slug": "",
                "repeat": 1,
                "item": "answer",
                "response": "B",
                "raw_output": '{"answer":"B"}',
                "age": None,
            }
        ]
    ).to_csv(target_csv, index=False)

    main(str(config_path), output_csv=str(target_csv))

    combined = pd.read_csv(target_csv)
    assert len(combined) == 2
    assert set(combined["vignette"]) == {"seatbelt", "casino_bad"}


def test_check_google_batch_status_holds_client_reference(monkeypatch):
    class DummyClient:
        def __init__(self, api_key):
            self.batches = self

        def get(self, name):
            return SimpleNamespace(state=SimpleNamespace(name="JOB_STATE_SUCCEEDED"))

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("law_norms_llms.process_results.genai.Client", DummyClient)

    status = check_google_batch_status("batches/123")
    assert status["status"] == "JOB_STATE_SUCCEEDED"


def test_process_batch_outputs_raises_for_terminal_google_state(tmp_path: Path, monkeypatch):
    class DummyCfg:
        experiment_name = "demo"

        def resolved_run_dir(self):
            return tmp_path / "runs"

    config_path = write_config(tmp_path, {"experiment_name": "demo"})
    (tmp_path / "runs").mkdir()

    monkeypatch.setattr(process_results, "load_experiment_config", lambda _: DummyCfg())
    monkeypatch.setattr(process_results, "load_execution_manifest", lambda _: {})
    monkeypatch.setattr(
        process_results,
        "load_batch_metadata",
        lambda _: {
            "batches": [
                {
                    "provider": "google",
                    "model": "gemini-3-flash-preview",
                    "model_id": "gemini-3-flash-preview",
                    "batch_id": "batches/123",
                }
            ]
        },
    )
    monkeypatch.setattr(
        process_results,
        "check_google_batch_status",
        lambda _: {"status": "JOB_STATE_FAILED", "error_message": "quota exceeded"},
    )
    monkeypatch.setattr(process_results, "create_results_dataframe", lambda _: pd.DataFrame())

    with pytest.raises(RuntimeError, match="JOB_STATE_FAILED: quota exceeded"):
        process_batch_outputs(str(config_path))
