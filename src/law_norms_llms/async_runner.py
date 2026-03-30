"""Run vignette experiments through LiteLLM without native provider batch APIs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from law_norms_llms.batch import run_repeats
from law_norms_llms.config import load_experiment_config
from law_norms_llms.expansion import build_execution_tasks
from law_norms_llms.providers import (
    async_reasoning_kwargs,
    build_openai_messages,
    model_id_for,
    model_output_tokens,
    model_reasoning,
    model_seed,
    model_temperature,
    provider_token_budget,
    reasoning_metadata,
)
from law_norms_llms.utils import sanitize_model_name, write_json


def save_execution_manifest(run_dir: Path, tasks) -> Path:
    """Persist the expanded execution manifest."""
    manifest = {task.request_stem: task.manifest_entry() for task in tasks}
    path = run_dir / "execution_manifest.json"
    write_json(path, manifest)
    return path


def main(config_path: str) -> None:
    """Run all vignette tasks asynchronously via LiteLLM."""
    config_file = Path(config_path).expanduser().resolve()
    cfg = load_experiment_config(config_file)
    tasks = build_execution_tasks(cfg)

    run_dir = cfg.resolved_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = save_execution_manifest(run_dir, tasks)
    logger.info(f"Execution manifest: {manifest_path}")

    for model_cfg in cfg.models:
        provider = model_cfg.provider
        model_name = model_cfg.name
        litellm_model = f"{provider}/{model_name}"
        model_temp = model_temperature(model_cfg, cfg.temperature)
        output_tokens = model_output_tokens(model_cfg, cfg.max_completion_tokens)
        model_seed_value = model_seed(model_cfg, cfg.seed)
        model_reasoning_cfg = model_reasoning(model_cfg)
        model_tokens = provider_token_budget(provider, output_tokens, model_reasoning_cfg)
        model_id = model_id_for(model_cfg, cfg.temperature)

        model_dir = run_dir / f"{provider}_{sanitize_model_name(model_id)}"
        model_dir.mkdir(parents=True, exist_ok=True)

        task_bar = tqdm(tasks, desc=f"{model_name}", unit="task")
        for task in task_bar:
            task_bar.set_postfix_str(task.request_stem, refresh=True)
            missing_runs = [
                repeat
                for repeat in range(1, cfg.repeats + 1)
                if not (model_dir / f"{task.request_stem}_repeat_{repeat:03d}.json").exists()
            ]
            if not missing_runs:
                continue

            messages = build_openai_messages(task.system_prompt, task.user_message)
            results = run_repeats(
                model=litellm_model,
                schema_name=task.schema_name,
                repeats=len(missing_runs),
                temperature=model_temp,
                max_tokens=output_tokens,
                chunk_size=cfg.concurrency,
                max_retries=cfg.max_retries,
                initial_backoff=cfg.initial_backoff,
                messages_template=messages,
                seed=model_seed_value,
                extra_request_kwargs=async_reasoning_kwargs(provider, output_tokens, model_reasoning_cfg),
            )

            for repeat_number, result in zip(missing_runs, results):
                payload = {
                    "provider": provider,
                    "model": model_name,
                    "model_id": model_id,
                    "temperature": model_temp,
                    "max_completion_tokens": model_tokens,
                    "output_tokens": output_tokens,
                    "reasoning": reasoning_metadata(model_reasoning_cfg),
                    "prompt_name": task.prompt_name,
                    "schema_name": task.schema_name,
                    "vignette_name": task.vignette_name,
                    "vignette_path": task.vignette_path,
                    "case_slug": task.case_slug,
                    "case_values": task.case_values,
                    "repeat": repeat_number,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "raw": result["raw"],
                    "json": result["json"],
                }
                write_json(model_dir / f"{task.request_stem}_repeat_{repeat_number:03d}.json", payload)


def cli() -> None:
    """Run the async runner CLI."""
    parser = argparse.ArgumentParser(description="Run vignette experiments asynchronously")
    parser.add_argument("config", help="Path to a vignette experiment YAML file")
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    cli()
