"""Legacy Streamlit dashboard for vignette experiment results."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from law_norms_llms.dashboard_data import (
    BASELINE_OPTION,
    RESPONSE_SCORES,
    available_vignette_options,
    format_vignette_label,
    prepare_results_frame,
    summarize_baseline_vignettes,
    summarize_threshold_group,
)
from law_norms_llms.utils import DATASETS_DIR


Y_TICKS = [round(-1.0 + 0.2 * index, 1) for index in range(11)]
Y_TICK_LABELS = [f"{tick:g}" for tick in Y_TICKS]
TICK_LABEL_SIZE = 8
MODEL_COLORS = list(plt.get_cmap("tab10").colors)
REFERENCE_LINE_COLOR = "#555555"
REFERENCE_LINE_STYLE = (0, (10, 6))
HUMAN_OPTION = "Humans"
MODEL_LABELS = {
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-2026-03-05": "GPT-5.4",
    "gemini-3-flash-preview": "Gemini 3 Flash Preview",
}
THRESHOLD_VARIANT_STYLES = {
    "casino_bad": {"label": "Negative", "color": "#b0b0b0", "linestyle": "--"},
    "casino_neutral": {"label": "Neutral", "color": "#000000", "linestyle": "-"},
    "casino_good": {"label": "Positive", "color": "#555555", "linestyle": "--"},
    "lorry_bad": {"label": "Negative", "color": "#b0b0b0", "linestyle": "--"},
    "lorry_neutral": {"label": "Neutral", "color": "#000000", "linestyle": "-"},
    "lorry_good": {"label": "Positive", "color": "#555555", "linestyle": "--"},
    "speeding_bad": {"label": "Negative", "color": "#b0b0b0", "linestyle": "--"},
    "speeding_neutral": {"label": "Neutral", "color": "#000000", "linestyle": "-"},
    "speeding_good": {"label": "Positive", "color": "#555555", "linestyle": "--"},
    "consent_16.5": {"label": "Man's Age: 16.5", "color": "#000000", "linestyle": "-"},
    "consent_28": {"label": "Man's Age: 28", "color": "#555555", "linestyle": "--"},
    "consent_40": {"label": "Man's Age: 40", "color": "#b0b0b0", "linestyle": "--"},
}
DATASET_PATHS = {
    "Baseline": DATASETS_DIR / "exp1.2.csv",
    "Roleplaying as Human": DATASETS_DIR / "exp1.3.csv",
    "Roleplaying as Human with Monetary Incentive": DATASETS_DIR / "exp1.4.csv",
}
HUMAN_DATA_PATH = DATASETS_DIR / "figure1.dta"
HUMAN_VIGNETTE_MAP = {
    "casino": "casino",
    "speeding": "speeding",
    "weight": "lorry",
    "sex": "consent",
}
HUMAN_VARIANT_MAP = {
    ("casino", "good"): "casino_good",
    ("casino", "neutral"): "casino_neutral",
    ("casino", "bad"): "casino_bad",
    ("speeding", "good"): "speeding_good",
    ("speeding", "neutral"): "speeding_neutral",
    ("speeding", "bad"): "speeding_bad",
    ("weight", "good"): "lorry_good",
    ("weight", "neutral"): "lorry_neutral",
    ("weight", "bad"): "lorry_bad",
    ("sex", "16"): "consent_16.5",
    ("sex", "28"): "consent_28",
    ("sex", "40"): "consent_40",
}
HUMAN_VARIANT_ORDER = {
    "casino": ["casino_good", "casino_neutral", "casino_bad"],
    "lorry": ["lorry_good", "lorry_neutral", "lorry_bad"],
    "speeding": ["speeding_good", "speeding_neutral", "speeding_bad"],
    "consent": ["consent_16.5", "consent_28", "consent_40"],
}
HUMAN_THRESHOLD_VALUES = {
    "casino": [
        "will turn 18 in seven days",
        "will turn 18 in five days",
        "will turn 18 in three days",
        "will turn 18 in one day",
        "turned 18 one day ago",
        "turned 18 three days ago",
        "turned 18 five days ago",
        "turned 18 seven days ago",
    ],
    "consent": [
        "15 years and 8 months",
        "15 years and 9 months",
        "15 years and 10 months",
        "15 years and 11 months",
        "16 years and 1 month",
        "16 years and 2 months",
        "16 years and 3 months",
        "16 years and 4 months",
    ],
    "lorry": [7.46, 7.47, 7.48, 7.49, 7.51, 7.52, 7.53, 7.54],
    "speeding": [66, 67, 68, 69, 71, 72, 73, 74],
}

plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    }
)


@st.cache_data(show_spinner=False)
def load_results(csv_path: str, score_mapping_key: tuple[tuple[str, int], ...]) -> pd.DataFrame:
    """Load and normalize a results CSV for the dashboard."""
    dataframe = pd.read_csv(csv_path)
    return prepare_results_frame(dataframe)


@st.cache_data(show_spinner=False)
def load_human_results(human_data_path: str) -> pd.DataFrame:
    """Load and normalize the human summary dataset."""
    path = Path(human_data_path)
    if not path.exists():
        return pd.DataFrame()

    dataframe = pd.read_stata(path)
    normalized = dataframe.copy()
    normalized["vignette"] = normalized["vignette"].astype(str)
    normalized["type"] = normalized["type"].astype(str)
    normalized["dashboard_vignette"] = normalized["vignette"].map(HUMAN_VIGNETTE_MAP)
    normalized["dashboard_variant"] = normalized.apply(
        lambda row: HUMAN_VARIANT_MAP.get((row["vignette"], row["type"])),
        axis=1,
    )
    normalized = normalized.dropna(subset=["dashboard_vignette", "dashboard_variant"]).copy()
    return normalized


def format_model_label(model: str) -> str:
    """Create a readable model label for buttons and legends."""
    if model in MODEL_LABELS:
        return MODEL_LABELS[model]
    return model.replace("-", " ").title()


def format_threshold_tick_label(value: object, threshold_column: str, vignette_group: str | None = None) -> str:
    """Wrap long categorical threshold labels to reduce width."""
    label = str(value)
    if threshold_column != "age":
        return label
    if "years and" in label:
        compact = label.replace(" years and ", "yr ").replace(" months", "m").replace(" month", "m")
        return compact
    if label.startswith("will turn 18 in "):
        suffix = label.removeprefix("will turn 18 in ")
        replacements = {
            "one day": "1 day",
            "three days": "3 days",
            "five days": "5 days",
            "seven days": "7 days",
        }
        return replacements.get(suffix, suffix)
    if label.startswith("turned 18 "):
        suffix = label.removeprefix("turned 18 ")
        replacements = {
            "one day ago": "-1 day",
            "three days ago": "-3 days",
            "five days ago": "-5 days",
            "seven days ago": "-7 days",
        }
        return replacements.get(suffix, f"-{suffix}")
    return label


def threshold_variant_style(variant: str) -> dict[str, str]:
    """Return the plotting style for a threshold variant."""
    return THRESHOLD_VARIANT_STYLES.get(
        variant,
        {"label": format_vignette_label(variant), "color": "#333333", "linestyle": "-"},
    )


def available_dataset_paths() -> dict[str, Path]:
    """Return the configured dashboard datasets that exist on disk."""
    return {
        label: path
        for label, path in DATASET_PATHS.items()
        if path.exists()
    }


def summarize_human_threshold_group(human_dataframe: pd.DataFrame, vignette_group: str) -> tuple[str, dict[str, pd.DataFrame]]:
    """Summarize the human dataset for one grouped threshold family."""
    filtered = human_dataframe.loc[human_dataframe["dashboard_vignette"] == vignette_group].copy()
    if filtered.empty:
        return "", {}

    threshold_column = "age" if vignette_group in {"casino", "consent"} else ("weight" if vignette_group == "lorry" else "speed")
    threshold_values = HUMAN_THRESHOLD_VALUES[vignette_group]
    sort_ascending = threshold_column != "age"
    summaries: dict[str, pd.DataFrame] = {}

    for variant in HUMAN_VARIANT_ORDER[vignette_group]:
        variant_rows = (
            filtered.loc[filtered["dashboard_variant"] == variant]
            .sort_values("threshold", ascending=sort_ascending)
            .reset_index(drop=True)
        )
        if variant_rows.empty:
            continue

        summary = variant_rows[["mean", "se", "upper", "lower", "n"]].copy()
        summary["threshold_value"] = threshold_values[: len(summary)]
        if threshold_column == "age":
            summary["plot_x"] = np.arange(len(summary), dtype=float)
        else:
            summary["plot_x"] = pd.to_numeric(pd.Series(threshold_values[: len(summary)]), errors="coerce")
        summary["plot_label"] = summary["threshold_value"].astype(str)
        summaries[variant] = summary

    return threshold_column, summaries


def build_baseline_figure(dataframe: pd.DataFrame, selected_models: list[str]) -> plt.Figure:
    """Create the baseline vignette bar chart."""
    fig, ax = plt.subplots(figsize=(6.4, 3.55))
    first_summary = summarize_baseline_vignettes(dataframe.loc[dataframe["model"] == selected_models[0]])
    vignette_order = first_summary["vignette"].tolist()
    labels = [format_vignette_label(vignette) for vignette in vignette_order]
    x_positions = np.arange(len(vignette_order), dtype=float)
    bar_width = min(0.8 / max(len(selected_models), 1), 0.35)

    for index, model in enumerate(selected_models):
        summary = summarize_baseline_vignettes(dataframe.loc[dataframe["model"] == model]).set_index("vignette")
        summary = summary.reindex(vignette_order)
        means = summary["mean"].to_numpy(dtype=float)
        lower_errors = means - summary["ci_lower"].to_numpy(dtype=float)
        upper_errors = summary["ci_upper"].to_numpy(dtype=float) - means
        offsets = x_positions + (index - (len(selected_models) - 1) / 2.0) * bar_width
        ax.bar(
            offsets,
            means,
            width=bar_width,
            yerr=[lower_errors, upper_errors],
            capsize=4,
            color=MODEL_COLORS[index % len(MODEL_COLORS)],
            edgecolor="#1f1f1f",
            label=format_model_label(model),
        )

    ax.set_title("Filler Vignettes", pad=25)
    ax.set_xlabel("Vignette")
    ax.set_ylabel("Mean appropriateness")
    ax.set_ylim(-1.1, 1.1)
    ax.set_yticks(Y_TICKS, Y_TICK_LABELS)
    ax.set_xticks(x_positions, labels)
    ax.axhline(0, linestyle="--", linewidth=1.2, color="#666666", alpha=0.8)
    ax.tick_params(axis="x", rotation=0, labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        loc="best",
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout()
    return fig


def build_threshold_figure(
    dataframe: pd.DataFrame,
    human_dataframe: pd.DataFrame,
    selected_series: list[str],
    vignette_group: str,
) -> plt.Figure:
    """Create the grouped threshold vignette line chart."""
    panel_count = len(selected_series)
    figure_width = 5.6 + max(panel_count - 1, 0) * 3.1
    figure_height = 3.25 + max(panel_count - 1, 0) * 0.2
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(figure_width, figure_height),
        sharey=True,
        squeeze=False,
    )
    axes_list = axes[0]
    for axis, series_name in zip(axes_list, selected_series):
        if series_name == HUMAN_OPTION:
            threshold_column, summaries = summarize_human_threshold_group(human_dataframe, vignette_group)
        else:
            threshold_column, summaries = summarize_threshold_group(
                dataframe.loc[dataframe["model"] == series_name],
                vignette_group,
            )
        if not summaries:
            axis.set_visible(False)
            continue
        reference_summary = next(iter(summaries.values()))

        for variant, summary in summaries.items():
            style = threshold_variant_style(variant)
            if "upper" in summary.columns and "lower" in summary.columns:
                means = summary["mean"].to_numpy(dtype=float)
                y_errors = [
                    means - summary["lower"].to_numpy(dtype=float),
                    summary["upper"].to_numpy(dtype=float) - means,
                ]
            elif "se" in summary.columns:
                y_errors = summary["se"].to_numpy(dtype=float)
            else:
                y_errors = [
                    summary["mean"] - summary["ci_lower"],
                    summary["ci_upper"] - summary["mean"],
                ]
            axis.errorbar(
                summary["plot_x"],
                summary["mean"],
                yerr=y_errors,
                fmt="o",
                linestyle=style["linestyle"],
                linewidth=2,
                capsize=4,
                color=style["color"],
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                label=style["label"],
            )

        axis.set_title(HUMAN_OPTION if series_name == HUMAN_OPTION else format_model_label(series_name), pad=10)
        axis.set_xlabel(threshold_column.title())
        axis.set_ylim(-1.1, 1.1)
        axis.set_yticks(Y_TICKS, Y_TICK_LABELS)
        axis.axhline(0, linestyle="--", linewidth=1.2, color="#666666", alpha=0.8)
        center_x = float((reference_summary["plot_x"].min() + reference_summary["plot_x"].max()) / 2.0)
        axis.axvline(center_x, linestyle=REFERENCE_LINE_STYLE, linewidth=1.3, color=REFERENCE_LINE_COLOR, alpha=0.9)
        axis.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
        axis.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
        axis.tick_params(axis="y", labelleft=True)
        axis.grid(alpha=0.25)

        if pd.to_numeric(reference_summary["threshold_value"], errors="coerce").isna().any():
            wrapped_labels = [
                format_threshold_tick_label(value, threshold_column, vignette_group)
                for value in reference_summary["threshold_value"]
            ]
            axis.set_xticks(reference_summary["plot_x"], wrapped_labels, rotation=0, ha="center")
            axis.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)

        if threshold_column == "age":
            axis.invert_xaxis()

        handles, labels = axis.get_legend_handles_labels()
        if handles and labels:
            axis.legend(
                handles,
                labels,
                loc="best",
                fontsize=8,
                frameon=False,
            )

    if any(axis.get_visible() for axis in axes_list):
        next(axis for axis in axes_list if axis.get_visible()).set_ylabel("Mean appropriateness")
    fig.subplots_adjust(wspace=0.12)
    fig.tight_layout()
    return fig


def render_threshold_figure(figure: plt.Figure, panel_count: int) -> None:
    """Render threshold figures with width that depends on the number of panels."""
    if panel_count <= 1:
        _, figure_column, _ = st.columns([1.2, 2.6, 1.2])
        with figure_column:
            st.pyplot(figure, use_container_width=False)
        return
    if panel_count == 2:
        _, figure_column, _ = st.columns([0.4, 3.8, 0.4])
        with figure_column:
            st.pyplot(figure, use_container_width=False)
        return
    st.pyplot(figure, use_container_width=False)


def toggle_model(model: str) -> None:
    """Toggle a model in the selected-model list."""
    selected_models = st.session_state.get("selected_models", [])
    if model in selected_models:
        st.session_state["selected_models"] = [item for item in selected_models if item != model]
    else:
        st.session_state["selected_models"] = [*selected_models, model]


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(page_title="Vignette Dashboard", layout="wide")
    st.markdown("<h1 style='text-align: center;'>Vignette Dashboard</h1>", unsafe_allow_html=True)

    _, plot_column, _ = st.columns([0.2, 4.6, 0.2], gap="small")

    dataset_paths = available_dataset_paths()
    if not dataset_paths:
        st.warning("No configured dashboard datasets were found.")
        return

    with plot_column:
        _, selector_column, _ = st.columns([0.7, 2.8, 0.7])
        with selector_column:
            selected_dataset = st.selectbox("Dataset", list(dataset_paths.keys()))
            csv_path = dataset_paths[selected_dataset]

    dataframe = load_results(str(csv_path), tuple(sorted(RESPONSE_SCORES.items())))
    if dataframe.empty:
        st.warning("No valid vignette responses were found in the selected CSV.")
        return
    human_dataframe = load_human_results(str(HUMAN_DATA_PATH))

    model_options = sorted(dataframe["model"].dropna().unique().tolist())
    selectable_series = [*model_options]
    human_available = not human_dataframe.empty
    if human_available:
        selectable_series.append(HUMAN_OPTION)
    selected_model_state = [
        model
        for model in st.session_state.get("selected_models", [])
        if model in selectable_series
    ]
    if not selected_model_state:
        selected_model_state = [model_options[0]]
    st.session_state["selected_models"] = selected_model_state

    with plot_column:
        _, selector_column, _ = st.columns([0.7, 2.8, 0.7])
        with selector_column:
            selected_vignette = st.selectbox(
                "Vignette",
                available_vignette_options(dataframe),
                format_func=format_vignette_label,
            )
            st.markdown("**Models**")
            for model in model_options:
                st.button(
                    format_model_label(model),
                    key=f"model_button_{selected_dataset}_{model}",
                    on_click=toggle_model,
                    args=(model,),
                    use_container_width=True,
                    type="primary" if model in st.session_state["selected_models"] else "secondary",
                )
            if selected_vignette != BASELINE_OPTION and human_available:
                st.button(
                    HUMAN_OPTION,
                    key=f"model_button_{selected_dataset}_{HUMAN_OPTION}",
                    on_click=toggle_model,
                    args=(HUMAN_OPTION,),
                    use_container_width=True,
                    type="primary" if HUMAN_OPTION in st.session_state["selected_models"] else "secondary",
                )

    if selected_vignette == BASELINE_OPTION:
        selected_models = [model for model in model_options if model in st.session_state["selected_models"]]
        if not selected_models:
            selected_models = [model_options[0]]
        if all(summarize_baseline_vignettes(dataframe.loc[dataframe["model"] == model]).empty for model in selected_models):
            st.warning("No baseline vignette rows were found for the selected models.")
            return

        figure = build_baseline_figure(dataframe, selected_models)
        with plot_column:
            _, figure_column, _ = st.columns([0.35, 3.5, 0.35])
            with figure_column:
                st.pyplot(figure, use_container_width=False)
        return

    selected_series = [
        series_name
        for series_name in [*model_options, HUMAN_OPTION]
        if series_name in st.session_state["selected_models"]
        and (series_name != HUMAN_OPTION or human_available)
    ]
    if not selected_series:
        selected_series = [model_options[0]]

    def _has_threshold_data(series_name: str) -> bool:
        if series_name == HUMAN_OPTION:
            return bool(summarize_human_threshold_group(human_dataframe, selected_vignette)[1])
        return bool(summarize_threshold_group(dataframe.loc[dataframe["model"] == series_name], selected_vignette)[1])

    if all(
        not _has_threshold_data(series_name)
        for series_name in selected_series
    ):
        st.warning("No threshold rows were found for this vignette and the selected models.")
        return

    figure = build_threshold_figure(dataframe, human_dataframe, selected_series, selected_vignette)
    with plot_column:
        render_threshold_figure(figure, len(selected_series))


if __name__ == "__main__":
    main()
