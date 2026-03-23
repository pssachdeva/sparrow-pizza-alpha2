import pytest

from law_norms_llms.human_regression import (
    DEFAULT_DATA_PATH,
    DEFAULT_RAW_DATA_PATH,
    VIGNETTE_LABELS,
    build_regression_table,
    build_raw_regression_table,
    load_human_data,
    load_raw_human_data,
)


def test_human_regression_reproduces_table1_coefficients():
    table = build_regression_table(load_human_data(DEFAULT_DATA_PATH))

    assert set(table["vignette"]) == set(VIGNETTE_LABELS.values())
    assert table.groupby("vignette").size().to_dict() == {
        "Age of Consent": 12,
        "Speeding": 12,
        "Casino": 12,
        "Lorry Weight": 12,
    }

    expected = {
        ("Age of Consent", "Illegal = 1"): -1.078181,
        ("Age of Consent", "Illegal x 40"): 0.673475,
        ("Speeding", "Clarity = negative"): -1.098303,
        ("Speeding", "Illegal x Distance"): 0.142071,
        ("Casino", "Illegal x positive"): 0.413316,
        ("Lorry Weight", "Illegal x Distance x negative"): 0.097103,
    }

    indexed = table.set_index(["vignette", "term_label"])
    for key, value in expected.items():
        assert indexed.loc[key, "coefficient"] == pytest.approx(value, abs=1e-6)


def test_human_regression_carries_original_participant_counts():
    table = build_regression_table(load_human_data(DEFAULT_DATA_PATH))
    counts = (
        table[["vignette", "participant_observations", "cell_observations"]]
        .drop_duplicates()
        .set_index("vignette")
    )

    assert counts.loc["Age of Consent", "participant_observations"] == 2909
    assert counts.loc["Speeding", "participant_observations"] == 2911
    assert counts.loc["Casino", "participant_observations"] == 2912
    assert counts.loc["Lorry Weight", "participant_observations"] == 2913
    assert (counts["cell_observations"] == 24).all()


def test_raw_human_regression_matches_paper_sample_counts_closely():
    table = build_raw_regression_table(load_raw_human_data(DEFAULT_RAW_DATA_PATH))
    counts = (
        table[["vignette", "participant_observations"]]
        .drop_duplicates()
        .set_index("vignette")["participant_observations"]
        .to_dict()
    )

    assert counts == {
        "Age of Consent": 2909,
        "Speeding": 2912,
        "Casino": 2913,
        "Lorry Weight": 2914,
    }


def test_raw_human_regression_reproduces_key_table1_coefficients():
    table = build_raw_regression_table(load_raw_human_data(DEFAULT_RAW_DATA_PATH))
    indexed = table.set_index(["vignette", "term_label"])

    expected = {
        ("Age of Consent", "Illegal = 1"): -1.088,
        ("Speeding", "Illegal x Distance"): 0.142,
        ("Casino", "Illegal = 1"): -1.264,
        ("Lorry Weight", "Illegal = 1"): -0.765,
    }

    for key, value in expected.items():
        assert indexed.loc[key, "coefficient"] == pytest.approx(value, abs=1e-3)
