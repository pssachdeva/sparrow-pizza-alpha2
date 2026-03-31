#!/usr/bin/env python3
"""Calculate threshold deltas for each vignette, model, prompt, and normative condition."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from law_norms_llms.dashboard_data import (
    adjacent_threshold_means,
    CONDITION_LABELS,
    THRESHOLD_VARIANT_ORDER,
    prepare_results_frame,
    threshold_distance_for_value,
)
from law_norms_llms.utils import DATASETS_DIR

MODEL_LABELS = {
    "claude-opus-4-6": "Claude Opus 4.6",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-2026-03-05": "GPT-5.4",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
}

PERSONA_LABELS = {
    "baseline": "Baseline",
    "roleplay_money": "Person",
    "roleplay_money_old": "Person (Old)",
    "roleplay_money_woman": "Woman",
    "roleplay_money_young": "Person (Young)",
    "persona_layperson": "Layperson",
    "persona_lawyer": "Lawyer",
    "persona_judge": "Judge",
}

VIGNETTE_TITLES = {
    "casino": "Casino",
    "consent": "Consent",
    "lorry": "Lorry",
    "speeding": "Speeding",
}

# Maps human .dta (vignette, type) -> our condition labels
HUMAN_VARIANT_MAP = {
    ("casino", "good"): ("casino", "Positive"),
    ("casino", "neutral"): ("casino", "Neutral"),
    ("casino", "bad"): ("casino", "Negative"),
    ("speeding", "good"): ("speeding", "Positive"),
    ("speeding", "neutral"): ("speeding", "Neutral"),
    ("speeding", "bad"): ("speeding", "Negative"),
    ("weight", "good"): ("lorry", "Positive"),
    ("weight", "neutral"): ("lorry", "Neutral"),
    ("weight", "bad"): ("lorry", "Negative"),
    ("sex", "16"): ("consent", "Neutral"),
    ("sex", "28"): ("consent", "Positive"),
    ("sex", "40"): ("consent", "Negative"),
}

HUMAN_DATA_PATH = DATASETS_DIR / "figure1.dta"


def compute_human_deltas() -> list[dict[str, object]]:
    """Compute threshold deltas from the human survey data."""
    if not HUMAN_DATA_PATH.exists():
        return []

    df = pd.read_stata(HUMAN_DATA_PATH)
    rows: list[dict[str, object]] = []

    for (vig, typ), group in df.groupby(["vignette", "type"]):
        mapping = HUMAN_VARIANT_MAP.get((str(vig), str(typ)))
        if mapping is None:
            continue
        vignette_group, condition = mapping

        means = adjacent_threshold_means(group, value_column="mean", distance_column="threshold1")
        if means is None:
            continue
        legal, illegal = means

        rows.append({
            "Model": "Humans",
            "Persona": "",
            "Vignette": vignette_group,
            "Condition": condition,
            "Delta": round(float(legal - illegal), 4),
        })

    return rows


def compute_deltas(csv_path: Path) -> pd.DataFrame:
    """Compute a single wide delta table (roleplay_money persona only, all vignettes)."""
    HIDDEN_MODELS = {"claude-sonnet-4-6"}
    raw = pd.read_csv(csv_path)
    df = prepare_results_frame(raw)
    df = df.loc[~df["model"].isin(HIDDEN_MODELS)]

    rows: list[dict[str, object]] = compute_human_deltas()

    for (model, prompt), group in df.groupby(["model", "prompt"]):
        for vignette_group, variants in THRESHOLD_VARIANT_ORDER.items():
            for variant in variants:
                vf = group.loc[group["vignette"] == variant].copy()
                vf = vf.dropna(subset=["threshold_value"])
                if vf.empty:
                    continue

                vf["threshold_distance"] = vf["threshold_value"].map(
                    lambda value, _variant=variant: threshold_distance_for_value(_variant, value)
                )
                means = adjacent_threshold_means(
                    vf,
                    value_column="response_score",
                    distance_column="threshold_distance",
                )
                if means is None:
                    continue
                legal_avg, illegal_avg = means

                rows.append({
                    "Model": MODEL_LABELS.get(model, model),
                    "Persona": PERSONA_LABELS.get(prompt, prompt),
                    "Vignette": vignette_group,
                    "Condition": CONDITION_LABELS.get(variant, variant),
                    "Delta": round(legal_avg - illegal_avg, 4),
                })

    long = pd.DataFrame(rows)
    if long.empty:
        return pd.DataFrame()

    # Keep only roleplay_money persona (and humans)
    long = long.loc[long["Persona"].isin(["Person", ""])].copy()
    long = long.drop(columns="Persona")

    # Title-case the vignette names
    long["Vignette"] = long["Vignette"].map(lambda v: VIGNETTE_TITLES.get(v, v.title()))

    wide = long.pivot_table(index=["Model", "Vignette"], columns="Condition", values="Delta", aggfunc="first").reset_index()
    wide.columns.name = None

    for cond in ("Positive", "Negative"):
        col = f"% Diff ({cond})"
        wide[col] = wide.apply(
            lambda r, _c=cond: round((r[_c] - r["Neutral"]) / abs(r["Neutral"]) * 100, 2)
            if pd.notna(r.get("Neutral")) and r.get("Neutral") != 0 and pd.notna(r.get(_c))
            else None,
            axis=1,
        )

    col_order = ["Vignette", "Model",
                 "Neutral", "Positive", "% Diff (Positive)",
                 "Negative", "% Diff (Negative)"]
    col_order = [c for c in col_order if c in wide.columns]
    wide = wide[col_order]

    # Sort by vignette, then Humans first within each vignette
    wide["_sort"] = wide["Model"].apply(lambda m: 0 if m == "Humans" else 1)
    wide = wide.sort_values(["Vignette", "_sort", "Model"]).drop(columns="_sort").reset_index(drop=True)

    return wide


def main() -> None:
    parser = argparse.ArgumentParser(description="Print threshold deltas as markdown tables")
    parser.add_argument("csv", nargs="?", default="datasets/all_models.csv", help="Path to results CSV")
    args = parser.parse_args()

    table = compute_deltas(Path(args.csv))
    if table.empty:
        print("No data found.")
        return

    print("\n## Threshold Deltas\n")
    print(table.to_markdown(index=False))


if __name__ == "__main__":
    main()
