#!/usr/bin/env python3
"""Compare beta3 (clarity x illegal interaction) across humans and LLMs."""
from __future__ import annotations

import argparse

import pandas as pd
import statsmodels.formula.api as smf

from law_norms_llms.human_regression import (
    DEFAULT_RAW_DATA_PATH,
    FORMULA,
    NEGATIVE_ILLEGAL_TERM,
    POSITIVE_ILLEGAL_TERM,
    VIGNETTE_LABELS,
    VIGNETTE_ORDER,
    build_raw_long_frame,
    load_raw_human_data,
    raw_sample_mask,
    significance_stars,
)
from law_norms_llms.llm_regression import (
    load_results,
    prepare_llm_frame,
    fit_model_vignette_regression,
)
from law_norms_llms.utils import DATASETS_DIR

MODEL_LABELS = {
    "claude-opus-4-6": "Claude Opus 4.6",
    "gpt-5.4": "GPT-5.4",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
}

TARGET_MODELS = ["claude-opus-4-6", "gpt-5.4", "gemini-3.1-pro-preview"]

# beta3: the clarity x illegal interaction terms
BETA3_TERMS = [
    ("Positive", POSITIVE_ILLEGAL_TERM),
    ("Negative", NEGATIVE_ILLEGAL_TERM),
]


def extract_beta3(fit) -> dict[str, dict]:
    """Extract beta3 (clarity x illegal interaction) coefficients."""
    results = {}
    for label, term in BETA3_TERMS:
        results[label] = {
            "coef": float(fit.params[term]),
            "se": float(fit.bse[term]),
            "p": float(fit.pvalues[term]),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare beta3 (clarity x illegal) across humans and LLMs")
    parser.add_argument("csv", nargs="?", default=str(DATASETS_DIR / "all_models.csv"), help="LLM results CSV")
    parser.add_argument("--prompt", default="roleplay_money", help="Prompt condition to use for LLMs")
    args = parser.parse_args()

    # Fit humans
    raw = load_raw_human_data(str(DEFAULT_RAW_DATA_PATH))
    long_frame = build_raw_long_frame(raw.loc[raw_sample_mask(raw)].copy())

    # Prepare LLM data
    df = load_results(args.csv)
    df = df[df["prompt"] == args.prompt].copy()
    prepared = prepare_llm_frame(df)

    model_frames: dict[str, pd.DataFrame] = {}
    for model in TARGET_MODELS:
        for mk in prepared["model_key"].unique():
            if model in mk:
                model_frames[model] = prepared.loc[prepared["model_key"] == mk].copy()
                break

    # Build one table per vignette
    for vignette in VIGNETTE_ORDER:
        label = VIGNETTE_LABELS[vignette]

        # Human fit
        vf = long_frame.loc[long_frame["vignette"] == vignette].copy()
        if vf.empty:
            continue
        human_fit = smf.ols(FORMULA, data=vf).fit(cov_type="HC1")
        human_b3 = extract_beta3(human_fit)

        rows: list[dict[str, object]] = []

        # Human row
        row: dict[str, object] = {"Source": "Humans"}
        for cond in ("Positive", "Negative"):
            h = human_b3[cond]
            row[f"β3 {cond}"] = f"{h['coef']:.3f}{significance_stars(h['p'])}"
            row[f"{cond} SE"] = f"({h['se']:.3f})"
            row[f"{cond} diff"] = ""
        rows.append(row)

        # LLM rows
        for model in TARGET_MODELS:
            mf = model_frames.get(model)
            if mf is None:
                continue
            mvf = mf.loc[mf["vignette_group"] == vignette].copy()
            if mvf.empty:
                continue
            llm_fit = fit_model_vignette_regression(mvf)
            llm_b3 = extract_beta3(llm_fit)

            display = MODEL_LABELS[model]
            row = {"Source": display}
            for cond in ("Positive", "Negative"):
                m = llm_b3[cond]
                h = human_b3[cond]
                row[f"β3 {cond}"] = f"{m['coef']:.3f}{significance_stars(m['p'])}"
                row[f"{cond} SE"] = f"({m['se']:.3f})"
                row[f"{cond} diff"] = f"{m['coef'] - h['coef']:+.3f}"
            rows.append(row)

        table = pd.DataFrame(rows)
        print(f"\n## {label} — Prompt: {args.prompt}\n")
        print(table.to_markdown(index=False))


if __name__ == "__main__":
    main()
