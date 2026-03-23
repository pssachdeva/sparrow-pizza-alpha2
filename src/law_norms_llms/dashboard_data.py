"""Aggregation helpers for the vignette Streamlit dashboard."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from law_norms_llms.utils import DATASETS_DIR

BASELINE_OPTION = "Filler Vignettes"
DEFAULT_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_SAMPLES = 2000
RESPONSE_SCORES = {"A": 1.0, "B": 1.0 / 3.0, "C": -1.0 / 3.0, "D": -1.0}
THRESHOLD_COLUMN_BY_PREFIX = {
    "casino": "age",
    "consent": "age",
    "lorry": "weight",
    "speeding": "speed",
}
THRESHOLD_VARIANT_ORDER = {
    "casino": ["casino_good", "casino_neutral", "casino_bad"],
    "lorry": ["lorry_good", "lorry_neutral", "lorry_bad"],
    "speeding": ["speeding_good", "speeding_neutral", "speeding_bad"],
    "consent": ["consent_16.5", "consent_28", "consent_40"],
}
CONDITION_LABELS = {
    "casino_good": "Positive",
    "casino_neutral": "Neutral",
    "casino_bad": "Negative",
    "lorry_good": "Positive",
    "lorry_neutral": "Neutral",
    "lorry_bad": "Negative",
    "speeding_good": "Positive",
    "speeding_neutral": "Neutral",
    "speeding_bad": "Negative",
    "consent_28": "Positive",
    "consent_16.5": "Neutral",
    "consent_40": "Negative",
}
CONDITION_ORDER = {"Positive": 0, "Neutral": 1, "Negative": 2}
DELTA_TABLE_COLUMNS = ["Neutral", "Positive", "Negative"]
NUMBER_WORDS = {
    "one": 1,
    "three": 3,
    "five": 5,
    "seven": 7,
}


def default_results_csv() -> Path:
    """Return the default CSV to load for the dashboard."""
    preferred = DATASETS_DIR / "exp1.2.csv"
    if preferred.exists():
        return preferred

    matches = sorted(DATASETS_DIR.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No CSV files found under {DATASETS_DIR}")
    return matches[0]


def threshold_column_for_vignette(vignette: str) -> str | None:
    """Return the threshold column used by a vignette, if any."""
    for prefix, column in THRESHOLD_COLUMN_BY_PREFIX.items():
        if vignette == prefix or vignette.startswith(f"{prefix}_"):
            return column
    return None


def threshold_group_for_vignette(vignette: str) -> str | None:
    """Return the grouped threshold family for a vignette, if any."""
    for prefix in THRESHOLD_COLUMN_BY_PREFIX:
        if vignette == prefix or vignette.startswith(f"{prefix}_"):
            return prefix
    return None


def threshold_variants_for_group(dataframe: pd.DataFrame, vignette_group: str) -> list[str]:
    """Return ordered vignette variants for a grouped threshold family."""
    configured_order = THRESHOLD_VARIANT_ORDER.get(vignette_group, [])
    available = dataframe.loc[
        dataframe["vignette"].astype(str).map(lambda value: threshold_group_for_vignette(value) == vignette_group),
        "vignette",
    ].dropna().unique().tolist()
    ordered = [variant for variant in configured_order if variant in available]
    extras = sorted(variant for variant in available if variant not in ordered)
    return [*ordered, *extras]


def format_vignette_label(vignette: str) -> str:
    """Create a human-readable vignette label for the UI."""
    if vignette == BASELINE_OPTION:
        return vignette
    return vignette.replace("_", " ").title()


def condition_label_for_variant(vignette: str) -> str:
    """Map one threshold vignette variant to its condition label."""
    return CONDITION_LABELS.get(vignette, format_vignette_label(vignette))


def split_threshold_values(threshold_values: list[object]) -> tuple[list[object], list[object]]:
    """Split ordered threshold values into before- and after-threshold segments."""
    if len(threshold_values) < 2:
        return ([], [])
    split_index = 4 if len(threshold_values) >= 8 else len(threshold_values) // 2
    return (threshold_values[:split_index], threshold_values[-split_index:])


def _parse_numeric_like(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _threshold_sort_key(vignette: str, value: object) -> tuple[int, float | str]:
    numeric_value = _parse_numeric_like(value)
    if numeric_value is not None:
        return (0, numeric_value)

    text = str(value).strip().lower()
    base_name = vignette.split("_", 1)[0]

    if base_name == "casino":
        if "will turn" in text:
            match = re.search(r"in\s+(one|three|five|seven|\d+)\s+day(?:s)?", text)
            if match:
                token = match.group(1)
                number = NUMBER_WORDS[token] if token in NUMBER_WORDS else int(token)
                return (0, -float(number))
        if "turned" in text:
            match = re.search(r"(one|three|five|seven|\d+)\s+day(?:s)?\s+ago", text)
            if match:
                token = match.group(1)
                number = NUMBER_WORDS[token] if token in NUMBER_WORDS else int(token)
                return (0, float(number))

    if base_name == "consent":
        match = re.search(r"(\d+)\s+years?\s+and\s+(\d+)\s+months?", text)
        if match:
            years = int(match.group(1))
            months = int(match.group(2))
            return (0, float(years * 12 + months))

    return (1, text)


def prepare_results_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw vignette results into a plotting-ready dataframe."""
    required_columns = {"model", "vignette", "repeat", "response"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    prepared = dataframe.copy()
    if "item" in prepared.columns:
        prepared = prepared[prepared["item"] == "answer"].copy()

    prepared["response_score"] = (
        prepared["response"].astype(str).str.strip().str.upper().map(RESPONSE_SCORES)
    )
    prepared["repeat"] = pd.to_numeric(prepared["repeat"], errors="coerce")
    prepared = prepared.dropna(subset=["model", "vignette", "repeat", "response_score"]).copy()
    prepared["response_score"] = prepared["response_score"].astype(float)

    prepared["threshold_column"] = prepared["vignette"].astype(str).map(threshold_column_for_vignette)
    prepared["is_threshold_vignette"] = prepared["threshold_column"].notna()
    prepared["threshold_value"] = None

    for column in sorted(prepared["threshold_column"].dropna().unique()):
        if column not in prepared.columns:
            continue
        mask = prepared["threshold_column"] == column
        prepared.loc[mask, "threshold_value"] = prepared.loc[mask, column]

    return prepared


def available_vignette_options(dataframe: pd.DataFrame) -> list[str]:
    """Return selector options for the current model subset."""
    thresholded = sorted(
        {
            threshold_group_for_vignette(vignette)
            for vignette in dataframe.loc[dataframe["is_threshold_vignette"], "vignette"].dropna().unique().tolist()
        }
    )
    if (~dataframe["is_threshold_vignette"]).any():
        return [BASELINE_OPTION, *thresholded]
    return thresholded


def bootstrap_mean_ci(
    values: pd.Series | list[float] | np.ndarray,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return the sample mean and a bootstrap confidence interval."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return (np.nan, np.nan, np.nan)

    mean = float(array.mean())
    if array.size == 1 or np.allclose(array, array[0]):
        return (mean, mean, mean)

    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_bootstrap, array.size), replace=True)
    sample_means = samples.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(sample_means, [alpha, 1.0 - alpha])
    return (mean, float(lower), float(upper))


def summarize_baseline_vignettes(
    dataframe: pd.DataFrame,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> pd.DataFrame:
    """Summarize non-threshold vignettes for a bar plot."""
    baseline = dataframe.loc[~dataframe["is_threshold_vignette"]].copy()
    if baseline.empty:
        return pd.DataFrame(columns=["vignette", "mean", "ci_lower", "ci_upper", "n"])

    summaries: list[dict[str, float | str | int]] = []
    for offset, vignette in enumerate(sorted(baseline["vignette"].dropna().unique().tolist())):
        values = baseline.loc[baseline["vignette"] == vignette, "response_score"]
        mean, ci_lower, ci_upper = bootstrap_mean_ci(
            values,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed + offset,
        )
        summaries.append(
            {
                "vignette": vignette,
                "mean": mean,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n": int(values.shape[0]),
            }
        )

    return pd.DataFrame(summaries)


def summarize_threshold_vignette(
    dataframe: pd.DataFrame,
    vignette: str,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> tuple[str, pd.DataFrame]:
    """Summarize a single threshold vignette for a line plot."""
    filtered = dataframe.loc[dataframe["vignette"] == vignette].copy()
    if filtered.empty:
        return "", pd.DataFrame(columns=["threshold_value", "mean", "ci_lower", "ci_upper", "n", "plot_x", "plot_label"])

    threshold_column = threshold_column_for_vignette(vignette)
    if threshold_column is None:
        raise ValueError(f"Vignette '{vignette}' does not have a threshold column")

    filtered = filtered.dropna(subset=["threshold_value"]).copy()
    raw_values = filtered["threshold_value"]
    numeric_values = pd.to_numeric(raw_values, errors="coerce")
    is_numeric = numeric_values.notna().all()

    threshold_order = sorted(pd.unique(raw_values).tolist(), key=lambda value: _threshold_sort_key(vignette, value))

    summaries: list[dict[str, float | str | int]] = []
    for offset, threshold_value in enumerate(threshold_order):
        values = filtered.loc[filtered["threshold_value"] == threshold_value, "response_score"]
        mean, ci_lower, ci_upper = bootstrap_mean_ci(
            values,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed + offset,
        )
        summaries.append(
            {
                "threshold_value": threshold_value,
                "mean": mean,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n": int(values.shape[0]),
            }
        )

    summary = pd.DataFrame(summaries)
    if is_numeric:
        summary["plot_x"] = pd.to_numeric(summary["threshold_value"])
    else:
        summary["plot_x"] = np.arange(summary.shape[0], dtype=float)
    summary["plot_label"] = summary["threshold_value"].astype(str)
    return threshold_column, summary


def summarize_threshold_group(
    dataframe: pd.DataFrame,
    vignette_group: str,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> tuple[str, dict[str, pd.DataFrame]]:
    """Summarize all variants inside a grouped threshold family."""
    threshold_column = threshold_column_for_vignette(vignette_group)
    if threshold_column is None:
        raise ValueError(f"Vignette group '{vignette_group}' does not have a threshold column")

    summaries: dict[str, pd.DataFrame] = {}
    for offset, variant in enumerate(threshold_variants_for_group(dataframe, vignette_group)):
        _, summary = summarize_threshold_vignette(
            dataframe,
            variant,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed + 1000 * offset,
        )
        if not summary.empty:
            summaries[variant] = summary
    return threshold_column, summaries


def build_threshold_delta_table(
    dataframe: pd.DataFrame,
    vignette_group: str,
    selected_models: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize before/after-threshold averages for one threshold family.

    For the standard eight-point threshold sweeps, "before" uses the first four
    ordered threshold values and "after" uses the last four. Shorter sweeps fall
    back to splitting the ordered threshold values in half.
    """
    threshold_column = threshold_column_for_vignette(vignette_group)
    if threshold_column is None:
        raise ValueError(f"Vignette group '{vignette_group}' does not have a threshold column")

    working = dataframe.copy()
    if selected_models is not None:
        working = working.loc[working["model"].isin(selected_models)].copy()
    if working.empty:
        return pd.DataFrame(columns=["Model", "Condition", "Before Threshold Avg.", "After Threshold Avg.", "Delta"])

    rows: list[dict[str, object]] = []
    model_order = selected_models or sorted(working["model"].dropna().astype(str).unique().tolist())
    variants = threshold_variants_for_group(working, vignette_group)

    for model in model_order:
        model_frame = working.loc[working["model"] == model].copy()
        if model_frame.empty:
            continue

        for variant in variants:
            variant_frame = model_frame.loc[model_frame["vignette"] == variant].copy()
            if variant_frame.empty:
                continue

            variant_frame = variant_frame.dropna(subset=["threshold_value"]).copy()
            threshold_order = sorted(
                pd.unique(variant_frame["threshold_value"]).tolist(),
                key=lambda value: _threshold_sort_key(variant, value),
            )
            before_values, after_values = split_threshold_values(threshold_order)
            if not before_values or not after_values:
                continue

            before_avg = float(
                variant_frame.loc[variant_frame["threshold_value"].isin(before_values), "response_score"].mean()
            )
            after_avg = float(
                variant_frame.loc[variant_frame["threshold_value"].isin(after_values), "response_score"].mean()
            )
            rows.append(
                {
                    "Model": model,
                    "Condition": condition_label_for_variant(variant),
                    "Before Threshold Avg.": before_avg,
                    "After Threshold Avg.": after_avg,
                    "Delta": before_avg - after_avg,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Model", "Condition", "Before Threshold Avg.", "After Threshold Avg.", "Delta"])

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        by=["Model", "Condition"],
        key=lambda column: (
            column.map(CONDITION_ORDER).fillna(len(CONDITION_ORDER))
            if column.name == "Condition"
            else column
        ),
        kind="stable",
    ).reset_index(drop=True)
    return summary


def build_threshold_delta_pivot(delta_table: pd.DataFrame) -> pd.DataFrame:
    """Pivot one threshold delta table into a one-row display-friendly table."""
    if delta_table.empty:
        return pd.DataFrame(columns=DELTA_TABLE_COLUMNS, index=["Delta"])

    row = {column: np.nan for column in DELTA_TABLE_COLUMNS}
    for _, entry in delta_table.iterrows():
        condition = str(entry["Condition"])
        if condition in row:
            row[condition] = float(entry["Delta"])
    return pd.DataFrame([row], index=["Delta"])
