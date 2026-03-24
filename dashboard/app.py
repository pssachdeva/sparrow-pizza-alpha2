"""Streamlit dashboard for vignette experiment results with prompt selection."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app_legacy import (
    BASELINE_OPTION,
    HUMAN_OPTION,
    HUMAN_DATA_PATH,
    RESPONSE_SCORES,
    build_baseline_figure,
    build_threshold_figure,
    format_model_label,
    format_vignette_label,
    load_human_results,
    load_results,
    render_threshold_figure,
    summarize_human_threshold_group,
)
from law_norms_llms.dashboard_data import (
    available_vignette_options,
    build_threshold_delta_table,
    build_threshold_delta_pivot,
    condition_label_for_variant,
    summarize_baseline_vignettes,
    summarize_threshold_group,
)
from law_norms_llms.utils import DATASETS_DIR

PROMPT_LABELS = {
    "baseline": "Baseline",
    "roleplay_money": "Roleplaying Person, Money Incentive",
    "roleplay_money_old": "Roleplaying Person (Old), Money Incentive",
    "roleplay_money_woman": "Roleplaying Woman, Money Incentive",
    "roleplay_money_young": "Roleplaying Person (Young), Money Incentive",
}


def available_dataset_paths() -> dict[str, Path]:
    """Return all dataset CSVs available for the dashboard."""
    return {path.stem: path for path in sorted(DATASETS_DIR.glob("*.csv"))}


def available_prompt_options(dataframe: pd.DataFrame) -> list[str]:
    """Return ordered prompt choices present in one dataset."""
    if "prompt" not in dataframe.columns:
        return []
    prompts = sorted(
        dataframe["prompt"].dropna().astype(str).unique().tolist(),
        key=lambda prompt_name: format_prompt_label(prompt_name).lower(),
    )
    return prompts


def filter_for_prompt(dataframe: pd.DataFrame, prompt_name: str | None) -> pd.DataFrame:
    """Filter one dataset to a single prompt if available."""
    if not prompt_name or "prompt" not in dataframe.columns:
        return dataframe.copy()
    return dataframe.loc[dataframe["prompt"].astype(str) == prompt_name].copy()


def format_prompt_label(prompt_name: str) -> str:
    """Map one internal prompt id to a display label."""
    return PROMPT_LABELS.get(prompt_name, prompt_name.replace("_", " ").title())


def toggle_series(state_key: str, series_name: str) -> None:
    """Toggle one series in the dataset-scoped selection state."""
    selected = st.session_state.get(state_key, [])
    if series_name in selected:
        st.session_state[state_key] = [item for item in selected if item != series_name]
        return
    st.session_state[state_key] = [*selected, series_name]


def build_human_delta_pivot(human_dataframe: pd.DataFrame, vignette_group: str) -> pd.DataFrame:
    """Build a one-row human delta table for the current threshold family."""
    _, summaries = summarize_human_threshold_group(human_dataframe, vignette_group)
    rows: list[dict[str, object]] = []

    for variant, summary in summaries.items():
        before_rows, after_rows = summary.iloc[:4], summary.iloc[-4:]
        if before_rows.empty or after_rows.empty:
            continue
        before_avg = float(before_rows["mean"].mean())
        after_avg = float(after_rows["mean"].mean())
        rows.append(
            {
                "Condition": condition_label_for_variant(variant),
                "Delta": before_avg - after_avg,
            }
        )

    return build_threshold_delta_pivot(pd.DataFrame(rows))


def main() -> None:
    """Render the prompt-aware Streamlit dashboard."""
    st.set_page_config(page_title="Vignette Dashboard", layout="wide")
    st.markdown("<h1 style='text-align: center;'>Law and Norms in LLMs</h1>", unsafe_allow_html=True)

    _, plot_column, _ = st.columns([0.2, 4.6, 0.2], gap="small")

    dataset_paths = available_dataset_paths()
    if not dataset_paths:
        st.warning(f"No dataset CSVs were found in {DATASETS_DIR}.")
        return

    with plot_column:
        _, selector_column, _ = st.columns([0.7, 2.8, 0.7])
        with selector_column:
            selected_dataset = st.selectbox("Dataset", list(dataset_paths.keys()))

    dataframe = load_results(str(dataset_paths[selected_dataset]), tuple(sorted(RESPONSE_SCORES.items())))
    if dataframe.empty:
        st.warning("No valid vignette responses were found in the selected CSV.")
        return

    prompt_options = available_prompt_options(dataframe)
    selected_prompt = None

    with plot_column:
        _, selector_column, _ = st.columns([0.7, 2.8, 0.7])
        with selector_column:
            if prompt_options:
                selected_prompt = st.selectbox("Prompt", prompt_options, format_func=format_prompt_label)

    filtered = filter_for_prompt(dataframe, selected_prompt)
    if filtered.empty:
        st.warning("No rows were found for the selected prompt.")
        return

    human_dataframe = load_human_results(str(HUMAN_DATA_PATH))

    vignette_options = available_vignette_options(filtered)
    if not vignette_options:
        st.warning("No vignette rows were found for the selected dataset and prompt.")
        return

    with plot_column:
        _, selector_column, _ = st.columns([0.7, 2.8, 0.7])
        with selector_column:
            selected_vignette = st.selectbox(
                "Vignette",
                vignette_options,
                format_func=format_vignette_label,
            )

    model_options = sorted(filtered["model"].dropna().astype(str).unique().tolist())
    if not model_options:
        st.warning("No model rows were found for the selected dataset and prompt.")
        return

    human_available = not human_dataframe.empty and selected_vignette != BASELINE_OPTION
    selectable_series = [*model_options, *([HUMAN_OPTION] if human_available else [])]
    selection_key = f"selected_models::{selected_dataset}"
    stored_selection = st.session_state.get(selection_key)
    if stored_selection is None:
        stored_selection = [model_options[0]]
        st.session_state[selection_key] = stored_selection
    active_selection = [series_name for series_name in stored_selection if series_name in selectable_series]
    if not active_selection:
        active_selection = [model_options[0]]

    with plot_column:
        _, selector_column, _ = st.columns([0.7, 2.8, 0.7])
        with selector_column:
            st.markdown("**Models**")
            for model in model_options:
                st.button(
                    format_model_label(model),
                    key=f"model_button::{selected_dataset}::{model}",
                    on_click=toggle_series,
                    args=(selection_key, model),
                    use_container_width=True,
                    type="primary" if model in active_selection else "secondary",
                )
            if human_available:
                st.button(
                    HUMAN_OPTION,
                    key=f"model_button::{selected_dataset}::{HUMAN_OPTION}",
                    on_click=toggle_series,
                    args=(selection_key, HUMAN_OPTION),
                    use_container_width=True,
                    type="primary" if HUMAN_OPTION in active_selection else "secondary",
                )

    if selected_vignette == BASELINE_OPTION:
        selected_models = [model for model in active_selection if model in model_options] or [model_options[0]]
        if all(summarize_baseline_vignettes(filtered.loc[filtered["model"] == model]).empty for model in selected_models):
            st.warning("No baseline vignette rows were found for the selected models.")
            return

        figure = build_baseline_figure(filtered, selected_models)
        with plot_column:
            _, figure_column, _ = st.columns([0.35, 3.5, 0.35])
            with figure_column:
                st.pyplot(figure, use_container_width=False)
        return

    selected_series = [series_name for series_name in active_selection if series_name in selectable_series] or [model_options[0]]

    def _has_threshold_data(series_name: str) -> bool:
        if series_name == HUMAN_OPTION:
            return bool(summarize_human_threshold_group(human_dataframe, selected_vignette)[1])
        return bool(summarize_threshold_group(filtered.loc[filtered["model"] == series_name], selected_vignette)[1])

    if all(not _has_threshold_data(series_name) for series_name in selected_series):
        st.warning("No threshold rows were found for this vignette and the selected models.")
        return

    figure = build_threshold_figure(filtered, human_dataframe, selected_series, selected_vignette)
    with plot_column:
        render_threshold_figure(figure, len(selected_series))
        st.markdown("<h3 style='text-align: center;'>Threshold Deltas</h3>", unsafe_allow_html=True)
        combined_rows: list[pd.DataFrame] = []
        for series_name in selected_series:
            if series_name == HUMAN_OPTION:
                delta_pivot = build_human_delta_pivot(human_dataframe, selected_vignette)
                model_label = HUMAN_OPTION
            else:
                delta_table = build_threshold_delta_table(filtered, selected_vignette, selected_models=[series_name])
                delta_pivot = build_threshold_delta_pivot(delta_table)
                model_label = format_model_label(series_name)

            if delta_pivot.empty or delta_pivot.isna().all(axis=None):
                continue

            display_row = delta_pivot.copy()
            for column in display_row.columns:
                display_row[column] = display_row[column].map(
                    lambda value: round(float(value), 3) if pd.notna(value) else value
                )
            display_row.insert(0, "Model", model_label)
            combined_rows.append(display_row)

        if combined_rows:
            display_table = pd.concat(combined_rows, ignore_index=True)
            _, table_column, _ = st.columns([1.3, 1.8, 1.3])
            with table_column:
                st.table(display_table)


if __name__ == "__main__":
    main()
