from pathlib import Path

import pytest
import yaml

from law_norms_llms.config import load_experiment_config
from law_norms_llms.expansion import build_execution_tasks


def write_config(tmp_path: Path, payload: dict) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload))
    return config_path


def test_build_execution_tasks_expands_globs_and_thresholds(tmp_path: Path):
    system_path = tmp_path / "system.txt"
    system_path.write_text("Judge the vignette named {vignette_name}.")
    (tmp_path / "casino_bad.txt").write_text("Customer is {age}.")
    (tmp_path / "casino_good.txt").write_text("Visitor is {age}.")
    (tmp_path / "lorry_bad.txt").write_text("Load is {weight} tons.")

    config_path = write_config(
        tmp_path,
        {
            "experiment_name": "demo",
            "run_dir": str(tmp_path / "runs"),
            "prompts": [
                {
                    "name": "appropriateness",
                    "schema_name": "VignetteResponse",
                    "system_template_path": str(system_path),
                }
            ],
            "vignettes": [
                {
                    "path": str(tmp_path / "casino_*.txt"),
                    "thresholds": {"age": [17, 18]},
                },
                {
                    "path": str(tmp_path / "lorry_bad.txt"),
                    "thresholds": {"weight": [7.5]},
                },
            ],
            "models": [{"provider": "openai", "name": "gpt-5.1"}],
        },
    )

    cfg = load_experiment_config(config_path)
    tasks = build_execution_tasks(cfg)

    assert len(tasks) == 5
    assert {task.vignette_name for task in tasks} == {"casino_bad", "casino_good", "lorry_bad"}
    assert {task.case_slug for task in tasks if task.vignette_name == "casino_bad"} == {"age_17", "age_18"}
    assert any(task.user_message == "Load is 7.5 tons." for task in tasks)


def test_build_execution_tasks_fails_on_missing_placeholders(tmp_path: Path):
    system_path = tmp_path / "system.txt"
    system_path.write_text("Return JSON.")
    vignette_path = tmp_path / "case.txt"
    vignette_path.write_text("Customer is {age}.")

    config_path = write_config(
        tmp_path,
        {
            "experiment_name": "demo",
            "run_dir": str(tmp_path / "runs"),
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

    cfg = load_experiment_config(config_path)
    with pytest.raises(ValueError, match="missing placeholders"):
        build_execution_tasks(cfg)
