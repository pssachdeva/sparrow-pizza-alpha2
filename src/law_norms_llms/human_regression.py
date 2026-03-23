"""Reproduce the paper's Table 1 regression on grouped or respondent-level human data."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

from law_norms_llms.utils import DATASETS_DIR, resolve_path


DEFAULT_DATA_PATH = DATASETS_DIR / "figure1.dta"
DEFAULT_RAW_DATA_PATH = DATASETS_DIR / "experiment_final.dta"
CLARITY_ORDER = ["neutral", "negative", "positive"]
VIGNETTE_ORDER = ["sex", "speeding", "casino", "weight"]
VIGNETTE_LABELS = {
    "sex": "Age of Consent",
    "speeding": "Speeding",
    "casino": "Casino",
    "weight": "Lorry Weight",
}
FORMULA = """
outcome ~ C(clarity, Treatment(reference="neutral"))
    + illegal
    + C(clarity, Treatment(reference="neutral")):illegal
    + distance
    + C(clarity, Treatment(reference="neutral")):distance
    + illegal:distance
    + C(clarity, Treatment(reference="neutral")):illegal:distance
""".strip()

NEGATIVE_TERM = 'C(clarity, Treatment(reference="neutral"))[T.negative]'
POSITIVE_TERM = 'C(clarity, Treatment(reference="neutral"))[T.positive]'
NEGATIVE_ILLEGAL_TERM = f"{NEGATIVE_TERM}:illegal"
POSITIVE_ILLEGAL_TERM = f"{POSITIVE_TERM}:illegal"
NEGATIVE_DISTANCE_TERM = f"{NEGATIVE_TERM}:distance"
POSITIVE_DISTANCE_TERM = f"{POSITIVE_TERM}:distance"
NEGATIVE_ILLEGAL_DISTANCE_TERM = f"{NEGATIVE_TERM}:illegal:distance"
POSITIVE_ILLEGAL_DISTANCE_TERM = f"{POSITIVE_TERM}:illegal:distance"

TERM_ORDER = [
    NEGATIVE_TERM,
    POSITIVE_TERM,
    "illegal",
    NEGATIVE_ILLEGAL_TERM,
    POSITIVE_ILLEGAL_TERM,
    "distance",
    NEGATIVE_DISTANCE_TERM,
    POSITIVE_DISTANCE_TERM,
    "illegal:distance",
    NEGATIVE_ILLEGAL_DISTANCE_TERM,
    POSITIVE_ILLEGAL_DISTANCE_TERM,
    "Intercept",
]

TERM_LABELS = {
    "sex": {
        NEGATIVE_TERM: "Clarity = 40",
        POSITIVE_TERM: "Clarity = 28",
        "illegal": "Illegal = 1",
        NEGATIVE_ILLEGAL_TERM: "Illegal x 40",
        POSITIVE_ILLEGAL_TERM: "Illegal x 28",
        "distance": "Distance",
        NEGATIVE_DISTANCE_TERM: "Distance x 40",
        POSITIVE_DISTANCE_TERM: "Distance x 28",
        "illegal:distance": "Illegal x Distance",
        NEGATIVE_ILLEGAL_DISTANCE_TERM: "Illegal x Distance x 40",
        POSITIVE_ILLEGAL_DISTANCE_TERM: "Illegal x Distance x 28",
        "Intercept": "Constant",
    },
    "default": {
        NEGATIVE_TERM: "Clarity = negative",
        POSITIVE_TERM: "Clarity = positive",
        "illegal": "Illegal = 1",
        NEGATIVE_ILLEGAL_TERM: "Illegal x negative",
        POSITIVE_ILLEGAL_TERM: "Illegal x positive",
        "distance": "Distance",
        NEGATIVE_DISTANCE_TERM: "Distance x negative",
        POSITIVE_DISTANCE_TERM: "Distance x positive",
        "illegal:distance": "Illegal x Distance",
        NEGATIVE_ILLEGAL_DISTANCE_TERM: "Illegal x Distance x negative",
        POSITIVE_ILLEGAL_DISTANCE_TERM: "Illegal x Distance x positive",
        "Intercept": "Constant",
    },
}

RAW_SCORE_MAP = {
    "Very inappropriate": -1.0,
    "Somewhat inappropriate": -1.0 / 3.0,
    "Somewhat appropriate": 1.0 / 3.0,
    "Very appropriate": 1.0,
}
SEX_THRESHOLD_MAP = {"158": -4, "159": -3, "1510": -2, "1511": -1, "161": 1, "162": 2, "163": 3, "164": 4}
SPEEDING_THRESHOLD_MAP = {"66": 4, "67": 3, "68": 2, "69": 1, "71": -1, "72": -2, "73": -3, "74": -4}
CASINO_THRESHOLD_MAP = {"177": -4, "175": -3, "173": -2, "171": -1, "181": 1, "183": 2, "185": 3, "187": 4}
WEIGHT_THRESHOLD_MAP = {"746": 4, "747": 3, "748": 2, "749": 1, "751": -1, "752": -2, "753": -3, "754": -4}
PAPER_ROW_ORDER = [
    "negative_term",
    "positive_term",
    "illegal",
    "negative_illegal",
    "positive_illegal",
    "distance",
    "negative_distance",
    "positive_distance",
    "illegal_distance",
    "negative_illegal_distance",
    "positive_illegal_distance",
    "constant",
    "observations",
]
PAPER_ROW_LABELS = {
    "negative_term": "= 40 / negative",
    "positive_term": "= 28 / positive",
    "illegal": "Illegal = 1 (I)",
    "negative_illegal": "I x 40 / negative",
    "positive_illegal": "I x 28 / positive",
    "distance": "Distance (D)",
    "negative_distance": "D x 40 / negative",
    "positive_distance": "D x 28 / positive",
    "illegal_distance": "I x D",
    "negative_illegal_distance": "I x D x 40 / negative",
    "positive_illegal_distance": "I x D x 28 / positive",
    "constant": "Constant",
    "observations": "Observations",
}
PAPER_ROW_KEYS_BY_TERM = {
    NEGATIVE_TERM: "negative_term",
    POSITIVE_TERM: "positive_term",
    "illegal": "illegal",
    NEGATIVE_ILLEGAL_TERM: "negative_illegal",
    POSITIVE_ILLEGAL_TERM: "positive_illegal",
    "distance": "distance",
    NEGATIVE_DISTANCE_TERM: "negative_distance",
    POSITIVE_DISTANCE_TERM: "positive_distance",
    "illegal:distance": "illegal_distance",
    NEGATIVE_ILLEGAL_DISTANCE_TERM: "negative_illegal_distance",
    POSITIVE_ILLEGAL_DISTANCE_TERM: "positive_illegal_distance",
    "Intercept": "constant",
}


def load_human_data(data_path: str | Path) -> pd.DataFrame:
    """Load the grouped human Figure 1 data."""
    resolved = resolve_path(data_path)
    df = pd.read_stata(resolved)
    required_columns = {"threshold1", "mean", "n", "vignette", "type"}
    missing = required_columns - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Grouped Figure 1 data is missing required columns: {missing_list}")
    return df


def load_raw_human_data(data_path: str | Path) -> pd.DataFrame:
    """Load the respondent-level experiment file."""
    resolved = resolve_path(data_path)
    return pd.read_stata(resolved)


def map_clarity(vignette: str, vignette_frame: pd.DataFrame) -> pd.Series:
    """Normalize vignette-specific clarity conditions into a shared coding."""
    if vignette == "sex":
        mapping = {"16": "neutral", "40": "negative", "28": "positive"}
    else:
        mapping = {"neutral": "neutral", "bad": "negative", "good": "positive"}
    clarity = vignette_frame["type"].astype(str).map(mapping)
    if clarity.isna().any():
        unknown = sorted(vignette_frame.loc[clarity.isna(), "type"].astype(str).unique())
        unknown_text = ", ".join(unknown)
        raise ValueError(f"Unexpected clarity values for vignette '{vignette}': {unknown_text}")
    return clarity


def prepare_vignette_data(df: pd.DataFrame, vignette: str) -> pd.DataFrame:
    """Create the regression-ready design for one vignette."""
    vignette_frame = df.loc[df["vignette"] == vignette].copy()
    if vignette_frame.empty:
        raise ValueError(f"No rows found for vignette '{vignette}'")
    vignette_frame["clarity"] = pd.Categorical(
        map_clarity(vignette, vignette_frame),
        categories=CLARITY_ORDER,
        ordered=True,
    )
    vignette_frame["illegal"] = (vignette_frame["threshold1"] < 0).astype(int)
    vignette_frame["distance"] = vignette_frame["threshold1"]
    vignette_frame["outcome"] = vignette_frame["mean"]
    return vignette_frame


def fit_grouped_vignette_regression(vignette_frame: pd.DataFrame):
    """Fit the paper's weighted OLS specification with HC1 robust errors."""
    return smf.wls(
        FORMULA,
        data=vignette_frame,
        weights=vignette_frame["n"],
    ).fit(cov_type="HC1")


def fit_raw_vignette_regression(vignette_frame: pd.DataFrame):
    """Fit the paper's OLS specification with HC1 robust errors on respondent data."""
    return smf.ols(FORMULA, data=vignette_frame).fit(cov_type="HC1")


def significance_stars(p_value: float) -> str:
    """Return conventional significance stars for a p-value."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def label_for_term(vignette: str, term: str) -> str:
    """Return a paper-style label for one fitted term."""
    labels = TERM_LABELS["sex"] if vignette == "sex" else TERM_LABELS["default"]
    return labels[term]


def build_regression_table(df: pd.DataFrame) -> pd.DataFrame:
    """Fit the paper's regression for each vignette and return a tidy coefficient table."""
    rows: list[dict[str, object]] = []

    for vignette in VIGNETTE_ORDER:
        vignette_frame = prepare_vignette_data(df, vignette)
        result = fit_grouped_vignette_regression(vignette_frame)
        participant_observations = int(vignette_frame["n"].sum())
        cell_observations = int(len(vignette_frame))

        for term in TERM_ORDER:
            rows.append(
                {
                    "vignette_key": vignette,
                    "vignette": VIGNETTE_LABELS[vignette],
                    "term": term,
                    "term_label": label_for_term(vignette, term),
                    "coefficient": float(result.params[term]),
                    "std_error": float(result.bse[term]),
                    "p_value": float(result.pvalues[term]),
                    "stars": significance_stars(float(result.pvalues[term])),
                    "participant_observations": participant_observations,
                    "cell_observations": cell_observations,
                    "rsquared": float(result.rsquared),
                }
            )

    return pd.DataFrame(rows)


def raw_sample_mask(df: pd.DataFrame) -> pd.Series:
    """Approximate the paper's final sample using the available Qualtrics screening fields."""
    return (
        (df["finished"] == 1)
        & (df["q_relevantidduplicate"] != 1)
        & (df["sc1"] == 1)
        & (df["q_relevantidfraudscore"].fillna(0) < 60)
    )


def parse_raw_column(column: str) -> tuple[str, str, int] | None:
    """Map a wide raw response column to vignette, clarity, and signed threshold distance."""
    if column.startswith("speeding_"):
        _, clarity_code, threshold_code = column.split("_")
        return "speeding", {"n": "neutral", "b": "negative", "g": "positive"}[clarity_code], SPEEDING_THRESHOLD_MAP[threshold_code]
    if column.startswith("casino_"):
        parts = column.split("_")
        if len(parts) == 2:
            return "casino", "neutral", CASINO_THRESHOLD_MAP[parts[1]]
        return "casino", {"b": "negative", "g": "positive"}[parts[1]], CASINO_THRESHOLD_MAP[parts[2]]
    if column.startswith("sex"):
        prefix, threshold_code = column.split("_")
        return "sex", {"16": "neutral", "40": "negative", "28": "positive"}[prefix[3:]], SEX_THRESHOLD_MAP[threshold_code]
    if column.startswith(("n_weight_", "g_weight_", "b_weight_")):
        clarity_code, threshold_code = column.split("_weight_")
        return "weight", {"n": "neutral", "b": "negative", "g": "positive"}[clarity_code], WEIGHT_THRESHOLD_MAP[threshold_code]
    return None


def build_raw_long_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape the respondent-level wide survey file into one row per rated vignette."""
    rows: list[dict[str, object]] = []

    for column in df.columns:
        parsed = parse_raw_column(column)
        if parsed is None:
            continue
        vignette, clarity, threshold1 = parsed
        scored = df[column].map(RAW_SCORE_MAP)
        for response_id, value in scored.dropna().items():
            rows.append(
                {
                    "respondent_index": response_id,
                    "vignette": vignette,
                    "clarity": clarity,
                    "illegal": int(threshold1 < 0),
                    "distance": threshold1,
                    "outcome": float(value),
                }
            )

    long_frame = pd.DataFrame(rows)
    long_frame["clarity"] = pd.Categorical(
        long_frame["clarity"],
        categories=CLARITY_ORDER,
        ordered=True,
    )
    return long_frame


def build_raw_regression_table(df: pd.DataFrame) -> pd.DataFrame:
    """Fit the paper's regression on respondent-level data and return a tidy coefficient table."""
    long_frame = build_raw_long_frame(df.loc[raw_sample_mask(df)].copy())
    rows: list[dict[str, object]] = []

    for vignette in VIGNETTE_ORDER:
        vignette_frame = long_frame.loc[long_frame["vignette"] == vignette].copy()
        result = fit_raw_vignette_regression(vignette_frame)

        for term in TERM_ORDER:
            rows.append(
                {
                    "vignette_key": vignette,
                    "vignette": VIGNETTE_LABELS[vignette],
                    "term": term,
                    "term_label": label_for_term(vignette, term),
                    "coefficient": float(result.params[term]),
                    "std_error": float(result.bse[term]),
                    "p_value": float(result.pvalues[term]),
                    "stars": significance_stars(float(result.pvalues[term])),
                    "participant_observations": int(len(vignette_frame)),
                    "cell_observations": int(vignette_frame["respondent_index"].nunique()),
                    "rsquared": float(result.rsquared),
                    "data_source": "raw",
                }
            )

    return pd.DataFrame(rows)


def render_summary(table: pd.DataFrame) -> str:
    """Render a compact human-readable summary of the fitted models."""
    lines = [
        "Paper Table 1 regression on grouped human Figure 1 data",
        "Note: coefficients reproduce the paper because the file contains cell means and counts.",
        "Note: HC1 standard errors are computed on the 24 weighted cells per vignette, so they are not the paper's subject-level robust SEs.",
        "",
    ]

    for vignette in VIGNETTE_ORDER:
        section = table.loc[table["vignette_key"] == vignette]
        if section.empty:
            continue
        lines.append(section["vignette"].iat[0])
        for _, row in section.iterrows():
            coefficient = f"{row['coefficient']:.3f}{row['stars']}"
            std_error = f"({row['std_error']:.3f})"
            lines.append(f"  {row['term_label']:<29} {coefficient:>10}  {std_error}")
        lines.append(
            "  "
            f"Responses={int(section['participant_observations'].iat[0])}, "
            f"Cells={int(section['cell_observations'].iat[0])}, "
            f"R^2={section['rsquared'].iat[0]:.3f}"
        )
        lines.append("")

    return "\n".join(lines).rstrip()


def render_paper_table(table: pd.DataFrame) -> str:
    """Render a compact Table 1-style coefficient matrix."""
    columns = [VIGNETTE_LABELS[vignette] for vignette in VIGNETTE_ORDER]
    cell_width = 18
    label_width = 24
    values: dict[str, dict[str, tuple[str, str]]] = {row_key: {} for row_key in PAPER_ROW_ORDER}

    for vignette in VIGNETTE_ORDER:
        vignette_label = VIGNETTE_LABELS[vignette]
        section = table.loc[table["vignette_key"] == vignette]
        for _, row in section.iterrows():
            row_key = PAPER_ROW_KEYS_BY_TERM[row["term"]]
            values[row_key][vignette_label] = (
                f"{row['coefficient']:.3f}{row['stars']}",
                f"({row['std_error']:.3f})",
            )
        values["observations"][vignette_label] = (str(int(section["participant_observations"].iat[0])), "")

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


def main(source: str, data_path: str, output_csv: str | None = None) -> None:
    """Run the grouped or respondent-level reproduction and print a compact summary."""
    if source == "grouped":
        table = build_regression_table(load_human_data(data_path))
        print(render_summary(table))
    else:
        table = build_raw_regression_table(load_raw_human_data(data_path))
        print("Paper Table 1 regression on respondent-level human data")
        print("Filter used: finished respondent, non-duplicate, passed comprehension screen, fraud score < 60")
        print("Note: this matches the published Age of Consent N exactly and the other three vignette Ns within 1 observation.")
        print()
        print(render_paper_table(table))

    if output_csv:
        output_path = resolve_path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
        print(f"\nSaved coefficient table to {output_path}")


def cli() -> None:
    """Run the human regression CLI."""
    parser = argparse.ArgumentParser(
        description="Reproduce the paper's Table 1 regression using grouped or respondent-level human data"
    )
    parser.add_argument(
        "--source",
        choices=["grouped", "raw"],
        default="raw",
        help="Which human-data source to use",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_RAW_DATA_PATH),
        help="Path to the Stata file matching the chosen source",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional path for a tidy CSV of coefficients and robust standard errors",
    )
    args = parser.parse_args()
    main(args.source, args.data, args.output_csv)


if __name__ == "__main__":
    cli()
