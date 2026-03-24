"""Process vignette batch outputs and normalize results into per-run JSON and CSV."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from anthropic import Anthropic
from google import genai
from loguru import logger
from openai import OpenAI

from law_norms_llms.config import load_experiment_config
from law_norms_llms.providers import extract_json
from law_norms_llms.schemas import make_refusal_response
from law_norms_llms.utils import datasets_dir_for_run_dir, sanitize_model_name, write_json

GOOGLE_BATCH_SUCCESS_STATES = {"JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED"}
GOOGLE_BATCH_TERMINAL_ERROR_STATES = {
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "BATCH_STATE_FAILED",
    "BATCH_STATE_CANCELLED",
    "BATCH_STATE_EXPIRED",
}


def parse_request_key(custom_id: str) -> tuple[str, int]:
    """Split a custom id into request stem and repeat number."""
    stem, repeat = custom_id.rsplit("_repeat_", 1)
    return stem, int(repeat)


def load_request_id_map(batch_entry: dict) -> dict[str, dict]:
    """Load the provider request id mapping for a submitted batch."""
    mapping_path = batch_entry.get("request_id_map_path")
    if not mapping_path:
        return {}
    return json.loads(Path(mapping_path).read_text())


def load_execution_manifest(run_dir: Path) -> dict[str, dict]:
    """Load the request manifest created before submission/execution."""
    return json.loads((run_dir / "execution_manifest.json").read_text())


def load_batch_metadata(run_dir: Path) -> dict:
    """Load submitted batch metadata."""
    return json.loads((run_dir / "batch_metadata.json").read_text())


def check_openai_batch_status(batch_id: str) -> dict:
    """Fetch OpenAI batch status."""
    batch = OpenAI().batches.retrieve(batch_id)
    return {
        "status": batch.status,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
    }


def download_openai_results(batch_id: str, run_dir: Path) -> Path:
    """Download OpenAI batch output."""
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise RuntimeError(f"Batch not completed yet. Status: {batch.status}")
    if not batch.output_file_id:
        raise RuntimeError("Batch completed but no output file available.")
    content = client.files.content(batch.output_file_id)
    output_path = run_dir / "batch_output.jsonl"
    output_path.write_bytes(content.content)
    return output_path


def check_anthropic_batch_status(batch_id: str) -> dict:
    """Fetch Anthropic batch status."""
    batch = Anthropic().messages.batches.retrieve(batch_id)
    return {"status": batch.processing_status}


def download_anthropic_results(batch_id: str, run_dir: Path) -> Path:
    """Download Anthropic batch output."""
    client = Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        raise RuntimeError(f"Batch not completed yet. Status: {batch.processing_status}")
    output_path = run_dir / "batch_output.jsonl"
    with open(output_path, "w") as handle:
        for result in client.messages.batches.results(batch_id):
            handle.write(json.dumps(result.model_dump()) + "\n")
    return output_path


def check_google_batch_status(batch_id: str) -> dict:
    """Fetch Gemini batch status."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable 'GEMINI_API_KEY' must be set")
    client = genai.Client(api_key=api_key)
    batch = client.batches.get(name=batch_id)
    state = getattr(batch, "state", None)
    status = getattr(state, "name", str(state)) if state is not None else "UNKNOWN"
    error = getattr(batch, "error", None)
    error_message = getattr(error, "message", None) if error is not None else None
    return {"status": status, "error_message": error_message}


def download_google_results(batch_id: str, run_dir: Path) -> Path:
    """Download Gemini batch output."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable 'GEMINI_API_KEY' must be set")

    client = genai.Client(api_key=api_key)
    batch = client.batches.get(name=batch_id)
    state = getattr(batch, "state", None)
    status = getattr(state, "name", str(state)) if state is not None else "UNKNOWN"
    if status not in GOOGLE_BATCH_SUCCESS_STATES:
        raise RuntimeError(f"Batch not completed yet. Status: {status}")

    file_name = batch.dest.file_name
    output_path = run_dir / "batch_output.jsonl"
    response = requests.get(
        f"https://generativelanguage.googleapis.com/v1beta/{file_name}:download?alt=media&key={api_key}"
    )
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def parse_and_save_batch_results(
    output_path: Path,
    run_dir: Path,
    batch_entry: dict,
    manifest: dict[str, dict],
    force: bool = False,
) -> None:
    """Parse a provider JSONL batch output into per-run result files."""
    provider = batch_entry["provider"]
    model_name = batch_entry["model"]
    model_id = batch_entry["model_id"]
    request_id_map = load_request_id_map(batch_entry)
    model_dir = run_dir / f"{provider}_{sanitize_model_name(model_id)}"
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path) as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            if provider == "openai":
                custom_id = payload["custom_id"]
                response = payload["response"]
                if response.get("status_code") != 200:
                    continue
                raw_text = response["body"]["choices"][0]["message"]["content"]
            elif provider == "anthropic":
                custom_id = payload["custom_id"]
                result = payload["result"]
                if result.get("type") == "errored":
                    continue
                if "message" not in result:
                    continue
                content_blocks = result["message"].get("content", [])
                raw_text = next(
                    (
                        block.get("text", "")
                        for block in content_blocks
                        if isinstance(block, dict) and block.get("type") == "text"
                    ),
                    "",
                )
                if not raw_text:
                    continue
            elif provider == "google":
                custom_id = payload.get("key", str(line_number))
                response = payload.get("response")
                if response is not None:
                    if "error" in response:
                        continue
                    candidates = response.get("candidates", [])
                else:
                    if "code" in payload and "message" in payload:
                        continue
                    if "error" in payload:
                        continue
                    candidates = payload.get("candidates", [])
                if not candidates:
                    continue
                parts = candidates[0].get("content", {}).get("parts", [])
                raw_text = parts[0].get("text", "") if parts else ""
            else:
                raise ValueError(f"Unknown provider: {provider}")

            if custom_id in request_id_map:
                request_stem = request_id_map[custom_id]["request_stem"]
                repeat = int(request_id_map[custom_id]["repeat"])
            else:
                request_stem, repeat = parse_request_key(custom_id)
            task = manifest[request_stem]
            output_file = model_dir / f"{request_stem}_repeat_{repeat:03d}.json"
            if output_file.exists() and not force:
                continue

            extracted = extract_json(raw_text)
            payload = {
                "provider": provider,
                "model": model_name,
                "model_id": model_id,
                "temperature": batch_entry["temperature"],
                "max_completion_tokens": batch_entry["max_completion_tokens"],
                "output_tokens": batch_entry.get("output_tokens"),
                "reasoning": batch_entry.get("reasoning"),
                "prompt_name": task["prompt_name"],
                "schema_name": task["schema_name"],
                "vignette_name": task["vignette_name"],
                "vignette_path": task["vignette_path"],
                "case_slug": task["case_slug"],
                "case_values": task["case_values"],
                "repeat": repeat,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw": raw_text,
                "json": extracted if extracted else make_refusal_response(),
            }
            write_json(output_file, payload)


def create_results_dataframe(run_dir: Path) -> pd.DataFrame:
    """Create a tidy dataframe from all per-run result files."""
    rows: list[dict] = []
    threshold_columns: set[str] = set()

    for model_dir in sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name != "__pycache__"):
        result_files = sorted(model_dir.glob("*_repeat_*.json"))
        for result_file in result_files:
            result = json.loads(result_file.read_text())
            case_values = result.get("case_values", {})
            threshold_columns.update(case_values)

            json_field = result.get("json")
            if isinstance(json_field, str):
                try:
                    parsed_json = json.loads(json_field)
                except json.JSONDecodeError:
                    parsed_json = {"answer": "REFUSAL"}
            elif isinstance(json_field, dict):
                parsed_json = json_field
            else:
                parsed_json = {"answer": "REFUSAL"}

            for item, response in parsed_json.items():
                row = {
                    "provider": result["provider"],
                    "model": result["model"],
                    "temperature": result["temperature"],
                    "prompt": result["prompt_name"],
                    "vignette": result["vignette_name"],
                    "vignette_path": result["vignette_path"],
                    "case_slug": result["case_slug"],
                    "repeat": result["repeat"],
                    "item": item,
                    "response": response,
                    "raw_output": result.get("raw", ""),
                }
                row.update(case_values)
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    columns = [
        "provider",
        "model",
        "temperature",
        "prompt",
        "vignette",
        "vignette_path",
        "case_slug",
        "repeat",
        "item",
        "response",
        "raw_output",
        *sorted(threshold_columns),
    ]
    return pd.DataFrame(rows)[columns]


def resolve_results_csv_path(cfg, run_dir: Path, output_csv: str | Path | None = None) -> Path:
    """Resolve where the normalized CSV should be written."""
    datasets_dir = datasets_dir_for_run_dir(run_dir)
    if output_csv is None:
        return datasets_dir / f"{cfg.experiment_name}.csv"

    csv_path = Path(output_csv).expanduser()
    if csv_path.is_absolute():
        return csv_path
    return datasets_dir / csv_path


def write_results_csv(dataframe: pd.DataFrame, destination: Path, append: bool = False) -> None:
    """Write normalized results to CSV, optionally appending to an existing file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if append and destination.exists():
        existing = pd.read_csv(destination)
        dataframe = pd.concat([existing, dataframe], ignore_index=True, sort=False)
    dataframe.to_csv(destination, index=False)


def should_append_results(output_csv: str | Path | None, append: bool) -> bool:
    """Choose append behavior, defaulting to safe appends for custom destinations."""
    if append:
        return True
    return output_csv is not None


def process_batch_outputs(
    config_path: str,
    force: bool = False,
    output_csv: str | Path | None = None,
    append: bool = False,
) -> None:
    """Download completed batch outputs, materialize per-run JSON, and save a CSV."""
    config_file = Path(config_path).expanduser().resolve()
    cfg = load_experiment_config(config_file)
    run_dir = cfg.resolved_run_dir()
    csv_path = resolve_results_csv_path(cfg, run_dir, output_csv=output_csv)
    append_results = should_append_results(output_csv, append)
    manifest = load_execution_manifest(run_dir)
    metadata = load_batch_metadata(run_dir)

    for batch_entry in metadata["batches"]:
        provider = batch_entry["provider"]
        model_id = batch_entry["model_id"]
        output_path = None
        status = None
        if provider == "openai":
            status = check_openai_batch_status(batch_entry["batch_id"])["status"]
            if status == "completed":
                output_path = download_openai_results(batch_entry["batch_id"], run_dir)
        elif provider == "anthropic":
            status = check_anthropic_batch_status(batch_entry["batch_id"])["status"]
            if status == "ended":
                output_path = download_anthropic_results(batch_entry["batch_id"], run_dir)
        elif provider == "google":
            status_payload = check_google_batch_status(batch_entry["batch_id"])
            status = status_payload["status"]
            if status in GOOGLE_BATCH_SUCCESS_STATES:
                output_path = download_google_results(batch_entry["batch_id"], run_dir)
            elif status in GOOGLE_BATCH_TERMINAL_ERROR_STATES:
                error_message = status_payload.get("error_message")
                detail = f": {error_message}" if error_message else ""
                raise RuntimeError(f"Batch for {provider}/{model_id} ended in terminal state {status}{detail}")
        else:
            raise ValueError(f"Unknown provider: {provider}")

        if output_path is None:
            logger.warning(f"Batch for {provider}/{model_id} is not complete yet. Status: {status}")
            continue

        model_output_path = run_dir / f"batch_output_{provider}_{sanitize_model_name(model_id)}.jsonl"
        if output_path != model_output_path:
            output_path.rename(model_output_path)
            output_path = model_output_path
        parse_and_save_batch_results(output_path, run_dir, batch_entry, manifest, force=force)

    dataframe = create_results_dataframe(run_dir)
    if not dataframe.empty:
        write_results_csv(dataframe, csv_path, append=append_results)
        logger.info(f"Extracted rows: {len(dataframe)}")
        logger.info(f"Saved normalized CSV: {csv_path.resolve()}")


def main(
    config_path: str,
    force: bool = False,
    output_csv: str | Path | None = None,
    append: bool = False,
) -> None:
    """Normalize whatever results are currently present for a vignette experiment."""
    cfg = load_experiment_config(Path(config_path).expanduser().resolve())
    run_dir = cfg.resolved_run_dir()
    csv_path = resolve_results_csv_path(cfg, run_dir, output_csv=output_csv)
    append_results = should_append_results(output_csv, append)

    batch_metadata_path = run_dir / "batch_metadata.json"
    if batch_metadata_path.exists():
        process_batch_outputs(config_path, force=force, output_csv=output_csv, append=append)
        return

    dataframe = create_results_dataframe(run_dir)
    if not dataframe.empty:
        write_results_csv(dataframe, csv_path, append=append_results)
        logger.info(f"Extracted rows: {len(dataframe)}")
        logger.info(f"Saved normalized CSV: {csv_path.resolve()}")


def cli() -> None:
    """Run the results processing CLI."""
    parser = argparse.ArgumentParser(description="Process vignette experiment results")
    parser.add_argument("config", help="Path to a vignette experiment YAML file")
    parser.add_argument("--force", action="store_true", help="Overwrite existing per-run JSON files from batch outputs")
    parser.add_argument(
        "--output-csv",
        help="Write the normalized CSV to this path instead of datasets/<experiment_name>.csv. Relative paths are resolved from the datasets directory associated with the run.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows to the destination CSV if it already exists. Custom --output-csv destinations append by default.",
    )
    args = parser.parse_args()
    main(args.config, force=args.force, output_csv=args.output_csv, append=args.append)


if __name__ == "__main__":
    cli()
