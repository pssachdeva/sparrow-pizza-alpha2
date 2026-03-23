"""Run the paper's Table 1 regression on LLM vignette results."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

from law_norms_llms.dashboard_data import RESPONSE_SCORES
from law_norms_llms.human_regression import (
    CLARITY_ORDER,
    FORMULA,
    PAPER_ROW_KEYS_BY_TERM,
    PAPER_ROW_LABELS,
    PAPER_ROW_ORDER,
    TERM_ORDER,
    VIGNETTE_LABELS,
    VIGNETTE_ORDER,
    label_for_term,
    resolve_path,
    significance_stars,
)
from law_norms_llms.utils import DATASETS_DIR


DEFAULT_RESULTS_PATH = DATASETS_DIR / "exp1.4.csv"
CONSENT_DISTANCE_MAP = {
    "15 years and 8 months": -4,
    "15 years and 9 months": -3,
    "15 years and 10 months": -2,
    "15 years and 11 months": -1,
    "16 years and 1 month": 1,
    "16 years and 2 months": 2,
    "16 years and 3 months": 3,
    "16 years and 4 months": 4,
}
CASINO_DISTANCE_MAP = {
    "will turn 18 in seven days": -4,
    "will turn 18 in five days": -3,
    "will turn 18 in three days": -2,
    "will turn 18 in one day": -1,
    "turned 18 one day ago": 1,
    "turned 18 three days ago": 2,
    "turned 18 five days ago": 3,
    "turned 18 seven days ago": 4,
}
SPEEDING_DISTANCE_MAP = {66.0: 4, 67.0: 3, 68.0: 2, 69.0: 1, 71.0: -1, 72.0: -2, 73.0: -3, 74.0: -4}
WEIGHT_DISTANCE_MAP = {7.46: 4, 7.47: 3, 7.48: 2, 7.49: 1, 7.51: -1, 7.52: -2, 7.53: -3, 7.54: -4}
VIGNETTE_METADATA = {
    "consent_16.5": {"group": "sex", "clarity": "neutral"},
    "consent_28": {"group": "sex", "clarity": "positive"},
    "consent_40": {"group": "sex", "clarity": "negative"},
    "speeding_neutral": {"group": "speeding", "clarity": "neutral"},
    "speeding_good": {"group": "speeding", "clarity": "positive"},
    "speeding_bad": {"group": "speeding", "clarity": "negative"},
    "casino_neutral": {"group": "casino", "clarity": "neutral"},
    "casino_good": {"group": "casino", "clarity": "positive"},
    "casino_bad": {"group": "casino", "clarity": "negative"},
    "lorry_neutral": {"group": "weight", "clarity": "neutral"},
    "lorry_good": {"group": "weight", "clarity": "positive"},
    "lorry_bad": {"group": "weight", "clarity": "negative"},
}
REQUIRED_COLUMNS = {"provider", "model", "repeat", "vignette", "response"}
THRESHOLD_COLUMNS = {"sex": "age", "casino": "age", "speeding": "speed", "weight": "weight"}


def load_results(path: str | Path) -> pd.DataFrame:
    """Load an LLM results CSV and validate the required columns."""
    resolved = resolve_path(path)
    df = pd.read_csv(resolved)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Results file is missing required columns: {missing_text}")
    return df


def normalize_threshold_value(group: str, value: object) -> int:
    """Convert one threshold value into signed distance from the legal threshold."""
    if group == "sex":
        key = str(value).strip()
        if key not in CONSENT_DISTANCE_MAP:
            raise ValueError(f"Unexpected consent age value: {value}")
        return CONSENT_DISTANCE_MAP[key]
    if group == "casino":
        key = str(value).strip()
        if key not in CASINO_DISTANCE_MAP:
            raise ValueError(f"Unexpected casino age value: {value}")
        return CASINO_DISTANCE_MAP[key]
    if group == "speeding":
        key = float(value)
        if key not in SPEEDING_DISTANCE_MAP:
            raise ValueError(f"Unexpected speeding value: {value}")
        return SPEEDING_DISTANCE_MAP[key]
    if group == "weight":
        key = round(float(value), 2)
        if key not in WEIGHT_DISTANCE_MAP:
            raise ValueError(f"Unexpected lorry weight value: {value}")
        return WEIGHT_DISTANCE_MAP[key]
    raise ValueError(f"Unknown vignette group: {group}")


def prepare_llm_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Filter threshold vignettes and normalize them into one regression-ready long frame."""
    prepared = df.copy()
    if "item" in prepared.columns:
        prepared = prepared[prepared["item"] == "answer"].copy()

    prepared["response"] = prepared["response"].astype(str).str.strip().str.upper()
    prepared["outcome"] = prepared["response"].map(RESPONSE_SCORES)
    prepared = prepared.dropna(subset=["outcome"]).copy()

    prepared = prepared[prepared["vignette"].isin(VIGNETTE_METADATA)].copy()
    prepared["model_key"] = prepared["provider"].astype(str) + "/" + prepared["model"].astype(str)

    groups = prepared["vignette"].map(lambda value: VIGNETTE_METADATA[value]["group"])
    prepared["vignette_group"] = groups
    prepared["clarity"] = prepared["vignette"].map(lambda value: VIGNETTE_METADATA[value]["clarity"])

    distances: list[int] = []
    for _, row in prepared.iterrows():
        group = row["vignette_group"]
        threshold_column = THRESHOLD_COLUMNS[group]
        if threshold_column not in prepared.columns:
            raise ValueError(f"Results file is missing threshold column '{threshold_column}'")
        distances.append(normalize_threshold_value(group, row[threshold_column]))

    prepared["distance"] = distances
    prepared["illegal"] = (prepared["distance"] < 0).astype(int)
    prepared["clarity"] = pd.Categorical(prepared["clarity"], categories=CLARITY_ORDER, ordered=True)
    prepared["repeat"] = pd.to_numeric(prepared["repeat"], errors="coerce")
    prepared = prepared.dropna(subset=["repeat"]).copy()
    prepared["repeat"] = prepared["repeat"].astype(int)
    return prepared


def fit_model_vignette_regression(vignette_frame: pd.DataFrame):
    """Fit the Table 1 OLS specification with HC1 robust errors for one model and vignette."""
    return smf.ols(FORMULA, data=vignette_frame).fit(cov_type="HC1")


def build_llm_regression_table(df: pd.DataFrame) -> pd.DataFrame:
    """Fit the Table 1 regression separately for each model."""
    prepared = prepare_llm_frame(df)
    rows: list[dict[str, object]] = []

    for model_key in sorted(prepared["model_key"].unique()):
        model_frame = prepared.loc[prepared["model_key"] == model_key].copy()
        provider, model_name = model_key.split("/", 1)

        for vignette in VIGNETTE_ORDER:
            vignette_frame = model_frame.loc[model_frame["vignette_group"] == vignette].copy()
            if vignette_frame.empty:
                continue
            result = fit_model_vignette_regression(vignette_frame)

            for term in TERM_ORDER:
                rows.append(
                    {
                        "provider": provider,
                        "model": model_name,
                        "model_key": model_key,
                        "vignette_key": vignette,
                        "vignette": VIGNETTE_LABELS[vignette],
                        "term": term,
                        "term_label": label_for_term(vignette, term),
                        "coefficient": float(result.params[term]),
                        "std_error": float(result.bse[term]),
                        "p_value": float(result.pvalues[term]),
                        "stars": significance_stars(float(result.pvalues[term])),
                        "observations": int(len(vignette_frame)),
                        "repeats": int(vignette_frame["repeat"].nunique()),
                        "rsquared": float(result.rsquared),
                    }
                )

    return pd.DataFrame(rows)


def render_model_table(model_table: pd.DataFrame) -> str:
    """Render one model's results in a Table 1-style layout."""
    columns = [VIGNETTE_LABELS[vignette] for vignette in VIGNETTE_ORDER]
    cell_width = 18
    label_width = 24
    values: dict[str, dict[str, tuple[str, str]]] = {row_key: {} for row_key in PAPER_ROW_ORDER}

    for vignette in VIGNETTE_ORDER:
        vignette_label = VIGNETTE_LABELS[vignette]
        section = model_table.loc[model_table["vignette_key"] == vignette]
        if section.empty:
            continue
        for _, row in section.iterrows():
            row_key = PAPER_ROW_KEYS_BY_TERM[row["term"]]
            values[row_key][vignette_label] = (
                f"{row['coefficient']:.3f}{row['stars']}",
                f"({row['std_error']:.3f})",
            )
        values["observations"][vignette_label] = (str(int(section["observations"].iat[0])), "")

    lines = []
    header = f"{'Term':<{label_width}}" + "".join(f"{column:>{cell_width}}" for column in columns)
    lines.append(header)

    for row_key in PAPER_ROW_ORDER:
        label = PAPER_ROW_LABELS[row_key]
        coef_line = f"{label:<{label_width}}"
        se_line = " " * label_width
        for column in columns:
            coefficient, std_error = values[row_key].get(column, ("", ""))
            coef_line += f"{coefficient:>{cell_width}}"
            se_line += f"{std_error:>{cell_width}}"
        lines.append(coef_line.rstrip())
        if row_key != "observations":
            lines.append(se_line.rstrip())

    return "\n".join(lines)


def render_all_models(table: pd.DataFrame) -> str:
    """Render all model-specific tables."""
    blocks = []
    for model_key in sorted(table["model_key"].unique()):
        model_table = table.loc[table["model_key"] == model_key].copy()
        blocks.append(model_key)
        repeats = int(model_table["repeats"].max())
        blocks.append(f"Treating repeats as respondents: {repeats}")
        blocks.append(render_model_table(model_table))
    return "\n\n".join(blocks)


def main(data_path: str, output_csv: str | None = None) -> None:
    """Run the Table 1 regression separately for each model in an LLM results CSV."""
    table = build_llm_regression_table(load_results(data_path))
    print(render_all_models(table))

    if output_csv:
        output_path = resolve_path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
        print(f"\nSaved coefficient table to {output_path}")


def cli() -> None:
    """Run the LLM regression CLI."""
    parser = argparse.ArgumentParser(
        description="Run the paper's Table 1 regression separately for each model in a vignette results CSV"
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_RESULTS_PATH),
        help="Path to a vignette results CSV such as datasets/exp1.4.csv",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional path for a tidy CSV of fitted coefficients and robust standard errors",
    )
    args = parser.parse_args()
    main(args.data, output_csv=args.output_csv)


if __name__ == "__main__":
    cli()
