import pandas as pd

from law_norms_llms.dashboard_data import (
    BASELINE_OPTION,
    available_vignette_options,
    build_threshold_delta_table,
    build_threshold_delta_pivot,
    prepare_results_frame,
    summarize_baseline_vignettes,
    summarize_threshold_group,
    summarize_threshold_vignette,
)


def sample_results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "model-a", "vignette": "bribe", "repeat": 1, "item": "answer", "response": "A"},
            {"model": "model-a", "vignette": "bribe", "repeat": 2, "item": "answer", "response": "A"},
            {"model": "model-a", "vignette": "seatbelt", "repeat": 1, "item": "answer", "response": "D"},
            {"model": "model-a", "vignette": "seatbelt", "repeat": 2, "item": "answer", "response": "D"},
            {"model": "model-a", "vignette": "casino_bad", "repeat": 1, "item": "answer", "response": "B", "age": "later"},
            {"model": "model-a", "vignette": "casino_bad", "repeat": 2, "item": "answer", "response": "B", "age": "later"},
            {"model": "model-a", "vignette": "casino_bad", "repeat": 3, "item": "answer", "response": "C", "age": "now"},
            {"model": "model-a", "vignette": "casino_bad", "repeat": 4, "item": "answer", "response": "C", "age": "now"},
            {"model": "model-a", "vignette": "lorry_bad", "repeat": 1, "item": "answer", "response": "A", "weight": 7.52},
            {"model": "model-a", "vignette": "lorry_bad", "repeat": 2, "item": "answer", "response": "A", "weight": 7.52},
            {"model": "model-a", "vignette": "lorry_bad", "repeat": 3, "item": "answer", "response": "D", "weight": 7.46},
            {"model": "model-a", "vignette": "lorry_bad", "repeat": 4, "item": "answer", "response": "D", "weight": 7.46},
            {"model": "model-a", "vignette": "bribe", "repeat": 3, "item": "other", "response": "D"},
        ]
    )


def test_prepare_results_frame_maps_letters_and_thresholds():
    prepared = prepare_results_frame(sample_results_frame())

    assert set(prepared["response_score"].unique()) == {-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0}
    assert prepared["item"].eq("answer").all()

    casino_rows = prepared.loc[prepared["vignette"] == "casino_bad"]
    assert casino_rows["threshold_column"].eq("age").all()
    assert casino_rows["threshold_value"].tolist() == ["later", "later", "now", "now"]


def test_available_vignette_options_groups_baseline_vignettes():
    prepared = prepare_results_frame(sample_results_frame())

    assert available_vignette_options(prepared) == [BASELINE_OPTION, "casino", "lorry"]


def test_summarize_baseline_vignettes_returns_means_and_cis():
    prepared = prepare_results_frame(sample_results_frame())
    summary = summarize_baseline_vignettes(prepared)
    summary = summary.set_index("vignette")

    assert summary.loc["bribe", "mean"] == 1.0
    assert summary.loc["bribe", "ci_lower"] == 1.0
    assert summary.loc["seatbelt", "mean"] == -1.0
    assert summary.loc["seatbelt", "ci_upper"] == -1.0


def test_summarize_threshold_vignette_orders_numeric_thresholds():
    prepared = prepare_results_frame(sample_results_frame())
    threshold_column, summary = summarize_threshold_vignette(prepared, "lorry_bad")

    assert threshold_column == "weight"
    assert summary["threshold_value"].tolist() == [7.46, 7.52]
    assert summary["mean"].tolist() == [-1.0, 1.0]


def test_summarize_threshold_vignette_orders_semantic_age_thresholds():
    dataframe = pd.DataFrame(
        [
            {"model": "model-a", "vignette": "casino_bad", "repeat": 1, "item": "answer", "response": "A", "age": "turned 18 one day ago"},
            {"model": "model-a", "vignette": "casino_bad", "repeat": 2, "item": "answer", "response": "A", "age": "turned 18 one day ago"},
            {"model": "model-a", "vignette": "casino_bad", "repeat": 3, "item": "answer", "response": "D", "age": "will turn 18 in seven days"},
            {"model": "model-a", "vignette": "casino_bad", "repeat": 4, "item": "answer", "response": "D", "age": "will turn 18 in seven days"},
        ]
    )

    prepared = prepare_results_frame(dataframe)
    _, summary = summarize_threshold_vignette(prepared, "casino_bad")

    assert summary["threshold_value"].tolist() == ["will turn 18 in seven days", "turned 18 one day ago"]


def test_summarize_threshold_group_collects_variants_under_one_family():
    prepared = prepare_results_frame(sample_results_frame())
    _, summaries = summarize_threshold_group(prepared, "lorry")

    assert list(summaries) == ["lorry_bad"]


def test_build_threshold_delta_table_summarizes_before_and_after_threshold():
    dataframe = pd.DataFrame(
        [
            *[
                {"model": "model-a", "vignette": "speeding_good", "repeat": idx, "item": "answer", "response": "D", "speed": speed}
                for idx, speed in enumerate([66, 67, 68, 69], start=1)
            ],
            *[
                {"model": "model-a", "vignette": "speeding_good", "repeat": idx + 4, "item": "answer", "response": "A", "speed": speed}
                for idx, speed in enumerate([71, 72, 73, 74], start=1)
            ],
            *[
                {"model": "model-a", "vignette": "speeding_neutral", "repeat": idx + 8, "item": "answer", "response": "C", "speed": speed}
                for idx, speed in enumerate([66, 67, 68, 69], start=1)
            ],
            *[
                {"model": "model-a", "vignette": "speeding_neutral", "repeat": idx + 12, "item": "answer", "response": "B", "speed": speed}
                for idx, speed in enumerate([71, 72, 73, 74], start=1)
            ],
            *[
                {"model": "model-a", "vignette": "speeding_bad", "repeat": idx + 16, "item": "answer", "response": "D", "speed": speed}
                for idx, speed in enumerate([66, 67, 68, 69], start=1)
            ],
            *[
                {"model": "model-a", "vignette": "speeding_bad", "repeat": idx + 20, "item": "answer", "response": "D", "speed": speed}
                for idx, speed in enumerate([71, 72, 73, 74], start=1)
            ],
        ]
    )

    prepared = prepare_results_frame(dataframe)
    summary = build_threshold_delta_table(prepared, "speeding", selected_models=["model-a"])

    assert summary["Condition"].tolist() == ["Positive", "Neutral", "Negative"]
    assert summary["Before Threshold Avg."].tolist() == [-1.0, -1.0 / 3.0, -1.0]
    assert summary["After Threshold Avg."].tolist() == [1.0, 1.0 / 3.0, -1.0]
    assert summary["Delta"].tolist() == [-2.0, -(2.0 / 3.0), 0.0]


def test_build_threshold_delta_pivot_orders_display_columns():
    delta_table = pd.DataFrame(
        [
            {"Condition": "Positive", "Delta": -0.5},
            {"Condition": "Neutral", "Delta": -0.25},
            {"Condition": "Negative", "Delta": 0.1},
        ]
    )

    pivot = build_threshold_delta_pivot(delta_table)

    assert pivot.columns.tolist() == ["Neutral", "Positive", "Negative"]
    assert pivot.loc["Delta", "Neutral"] == -0.25
    assert pivot.loc["Delta", "Positive"] == -0.5
    assert pivot.loc["Delta", "Negative"] == 0.1
