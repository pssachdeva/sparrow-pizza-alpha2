import json
from pathlib import Path

import yaml

from law_norms_llms import async_runner


def write_config(tmp_path: Path, payload: dict) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload))
    return config_path


def test_async_runner_writes_one_result_per_task_and_repeat(tmp_path: Path, monkeypatch):
    system_path = tmp_path / "system.txt"
    system_path.write_text("Return JSON.")
    (tmp_path / "casino_bad.txt").write_text("Customer is {age}.")
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
                    "path": str(tmp_path / "casino_*.txt"),
                    "thresholds": {"age": [17, 18]},
                }
            ],
            "models": [{"provider": "openrouter", "name": "qwen/qwen3-max"}],
        },
    )

    def fake_run_repeats(**kwargs):
        return [
            {"run_index": 0, "raw": '{"answer":"A"}', "json": '{"answer":"A"}'},
            {"run_index": 1, "raw": '{"answer":"A"}', "json": '{"answer":"A"}'},
        ]

    monkeypatch.setattr(async_runner, "run_repeats", fake_run_repeats)
    async_runner.main(str(config_path))

    model_dir = run_dir / "openrouter_qwen_qwen3-max"
    result_files = sorted(model_dir.glob("*_repeat_*.json"))
    assert len(result_files) == 4

    payload = json.loads(result_files[0].read_text())
    assert payload["prompt_name"] == "appropriateness"
    assert payload["case_values"]["age"] in {17, 18}
