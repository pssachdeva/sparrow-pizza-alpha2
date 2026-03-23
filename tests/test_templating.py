from law_norms_llms.templating import extract_placeholders, missing_placeholders, render_template


def test_extract_placeholders_ignores_escaped_braces():
    text = "Age {age}; literal braces {{ok}}."
    assert extract_placeholders(text) == {"age"}


def test_missing_placeholders_reports_only_absent_values():
    text = "Age {age} Weight {weight}"
    assert missing_placeholders(text, {"age"}) == {"weight"}


def test_render_template_substitutes_values():
    text = "The person is {age} years old."
    assert render_template(text, {"age": 17}) == "The person is 17 years old."
