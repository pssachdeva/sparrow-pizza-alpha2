import pytest

from law_norms_llms.llm_regression import (
    DEFAULT_RESULTS_PATH,
    build_llm_regression_table,
    load_results,
)


def test_llm_regression_uses_repeats_as_respondents():
    table = build_llm_regression_table(load_results(DEFAULT_RESULTS_PATH))
    counts = table[["model_key", "vignette", "observations", "repeats"]].drop_duplicates()

    assert len(counts) == 8
    assert counts["observations"].eq(720).all()
    assert counts["repeats"].eq(30).all()


def test_llm_regression_matches_expected_coefficients_for_exp14():
    table = build_llm_regression_table(load_results(DEFAULT_RESULTS_PATH))
    indexed = table.set_index(["model_key", "vignette", "term_label"])

    expected = {
        ("anthropic/claude-sonnet-4-6", "Age of Consent", "Illegal = 1"): -1.333333,
        ("openai/gpt-5.4-2026-03-05", "Speeding", "Illegal = 1"): -0.566667,
        ("anthropic/claude-sonnet-4-6", "Casino", "Illegal x positive"): 0.022222,
        ("openai/gpt-5.4-2026-03-05", "Lorry Weight", "Illegal = 1"): -0.911111,
    }

    for key, value in expected.items():
        assert indexed.loc[key, "coefficient"] == pytest.approx(value, abs=1e-6)
