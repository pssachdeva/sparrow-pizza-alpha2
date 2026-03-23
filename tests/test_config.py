from pathlib import Path

import pytest
import yaml

from law_norms_llms.config import load_experiment_config


def test_load_experiment_config_parses_threshold_maps(tmp_path: Path):
    system_path = tmp_path / "system.txt"
    system_path.write_text("Return JSON only.")
    vignette_path = tmp_path / "case.txt"
    vignette_path.write_text("A person is {age} years old.")
    run_dir = tmp_path / "out"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
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
                        "thresholds": {"age": [17, 18, 19]},
                    }
                ],
                "models": [{"provider": "openai", "name": "gpt-5.1"}],
            }
        )
    )

    cfg = load_experiment_config(config_path)
    assert cfg.experiment_name == "demo"
    assert cfg.prompts[0].schema_name == "VignetteResponse"
    assert cfg.vignettes[0].thresholds == {"age": [17, 18, 19]}
    assert cfg.resolved_run_dir() == run_dir


def test_load_experiment_config_parses_output_tokens_and_reasoning(tmp_path: Path):
    system_path = tmp_path / "system.txt"
    system_path.write_text("Return JSON only.")
    vignette_path = tmp_path / "case.txt"
    vignette_path.write_text("A person is {age} years old.")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "demo",
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
                    {
                        "provider": "google",
                        "name": "gemini-3-flash-preview",
                        "output_tokens": 80,
                        "reasoning": {
                            "effort": "minimal",
                            "tokens": 256,
                        },
                    }
                ],
            }
        )
    )

    cfg = load_experiment_config(config_path)
    assert cfg.models[0].output_tokens == 80
    assert cfg.models[0].reasoning.effort == "minimal"
    assert cfg.models[0].reasoning.tokens == 256


def test_load_experiment_config_rejects_legacy_output_dir(tmp_path: Path):
    system_path = tmp_path / "system.txt"
    system_path.write_text("Return JSON only.")
    vignette_path = tmp_path / "case.txt"
    vignette_path.write_text("A person is {age} years old.")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "demo",
                "output_dir": str(tmp_path / "runs"),
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
                "models": [{"provider": "openai", "name": "gpt-5.1"}],
            }
        )
    )

    with pytest.raises(Exception, match="output_dir"):
        load_experiment_config(config_path)
