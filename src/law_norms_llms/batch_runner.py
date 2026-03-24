"""Submit vignette experiments to provider-native batch APIs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from law_norms_llms.batch import create_anthropic_batch, create_google_batch, create_openai_batch
from law_norms_llms.config import load_experiment_config
from law_norms_llms.expansion import build_execution_tasks
from law_norms_llms.providers import (
    model_id_for,
    model_output_tokens,
    model_reasoning,
    model_seed,
    model_temperature,
    model_use_structured_output,
    provider_token_budget,
    reasoning_metadata,
)
from law_norms_llms.utils import sanitize_model_name, write_json


def save_execution_manifest(run_dir: Path, tasks) -> Path:
    """Persist the fully expanded execution manifest."""
    manifest = {task.request_stem: task.manifest_entry() for task in tasks}
    path = run_dir / "execution_manifest.json"
    write_json(path, manifest)
    return path


def save_request_id_map(run_dir: Path, provider: str, safe_model: str, request_id_map: dict) -> Path:
    """Persist provider-facing batch request ids to manifest task ids."""
    path = run_dir / f"request_id_map_{provider}_{safe_model}.json"
    write_json(path, request_id_map)
    return path


def save_batch_metadata(cfg, config_path: Path, run_dir: Path, batch_entries: list[dict]) -> Path:
    """Persist submitted batch metadata."""
    payload = {
        "experiment_name": cfg.experiment_name,
        "config_path": str(config_path),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "num_prompts": len(cfg.prompts),
        "num_vignette_groups": len(cfg.vignettes),
        "num_repeats": cfg.repeats,
        "run_dir": str(run_dir),
        "prompts": [
            {
                "name": prompt.name,
                "schema_name": prompt.schema_name,
                "system_template_path": str(prompt.system_template_path) if prompt.system_template_path else None,
            }
            for prompt in cfg.prompts
        ],
        "vignettes": [
            {
                "name": vignette.name,
                "path": vignette.path,
                "thresholds": vignette.thresholds,
            }
            for vignette in cfg.vignettes
        ],
        "batches": batch_entries,
    }
    path = run_dir / "batch_metadata.json"
    write_json(path, payload)
    return path


def main(config_path: str) -> None:
    """Submit one native batch per configured model."""
    config_file = Path(config_path).expanduser().resolve()
    cfg = load_experiment_config(config_file)
    tasks = build_execution_tasks(cfg)
    requests_per_model = len(tasks) * cfg.repeats
    total_requests = requests_per_model * len(cfg.models)

    run_dir = cfg.resolved_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = save_execution_manifest(run_dir, tasks)
    logger.info(f"Execution manifest: {manifest_path}")
    logger.info(f"Expanded tasks: {len(tasks)}")
    logger.info(f"Requests per model batch: {requests_per_model}")
    logger.info(f"Total requests across configured models: {total_requests}")

    batch_entries: list[dict] = []
    metadata_path = save_batch_metadata(cfg, config_file, run_dir, batch_entries)
    for model_cfg in cfg.models:
        provider = model_cfg.provider
        model_name = model_cfg.name
        model_temp = model_temperature(model_cfg, cfg.temperature)
        output_tokens = model_output_tokens(model_cfg, cfg.max_completion_tokens)
        model_seed_value = model_seed(model_cfg, cfg.seed)
        model_structured = model_use_structured_output(model_cfg, cfg.use_structured_output)
        model_reasoning_cfg = model_reasoning(model_cfg)
        model_tokens = provider_token_budget(provider, output_tokens, model_reasoning_cfg)
        model_id = model_id_for(model_cfg, cfg.temperature)
        safe_model = sanitize_model_name(model_id)
        batch_input_path = run_dir / f"batch_input_{provider}_{safe_model}.jsonl"

        if provider == "openai":
            batch_info = create_openai_batch(
                experiment_name=cfg.experiment_name,
                model_name=model_name,
                tasks=tasks,
                repeats=cfg.repeats,
                temperature=model_temp,
                output_tokens=output_tokens,
                use_structured_output=model_structured,
                batch_input_path=batch_input_path,
                seed=model_seed_value,
                reasoning=model_reasoning_cfg,
            )
        elif provider == "anthropic":
            batch_info = create_anthropic_batch(
                model_name=model_name,
                tasks=tasks,
                repeats=cfg.repeats,
                temperature=model_temp,
                output_tokens=output_tokens,
                batch_input_path=batch_input_path,
                reasoning=model_reasoning_cfg,
            )
        elif provider == "google":
            batch_info = create_google_batch(
                model_name=model_name,
                tasks=tasks,
                repeats=cfg.repeats,
                temperature=model_temp,
                output_tokens=output_tokens,
                use_structured_output=model_structured,
                batch_input_path=batch_input_path,
                reasoning=model_reasoning_cfg,
            )
        else:
            raise ValueError(f"Unsupported provider for native batch: {provider}")

        request_id_map = batch_info.pop("request_id_map", {})
        request_id_map_path = save_request_id_map(run_dir, provider, safe_model, request_id_map)
        batch_entries.append(
            {
                "provider": provider,
                "model": model_name,
                "model_id": model_id,
                "temperature": model_temp,
                "max_completion_tokens": model_tokens,
                "output_tokens": output_tokens,
                "reasoning": reasoning_metadata(model_reasoning_cfg),
                "request_id_map_path": str(request_id_map_path),
                **batch_info,
            }
        )
        metadata_path = save_batch_metadata(cfg, config_file, run_dir, batch_entries)

    logger.info(f"Saved batch metadata: {metadata_path}")


def cli() -> None:
    """Run the native batch submission CLI."""
    parser = argparse.ArgumentParser(description="Submit vignette experiments to batch APIs")
    parser.add_argument("config", help="Path to a vignette experiment YAML file")
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    cli()
