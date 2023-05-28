import pandas as pd
import pytest
from utils.text import (
    trim_motion_text_by_subtitle,
    trim_motion_text_by_leading_title,
    trim_whitespace,
    trim_linebreaks,
    trim_motion_text_by_proposed_decision,
    trim_leadning_motivation,
    set_empty_when_leading_date,
    delete_footer,
    prep_text,
)


@pytest.fixture
def sample_row():
    return pd.Series(
        {
            "text": "This is a sample text.",
            "subtitle": "This is",
            "title": "This is",
        }
    )


def test_trim_motion_text_by_subtitle(sample_row):
    result = trim_motion_text_by_subtitle(sample_row)
    assert result == "a sample text."


def test_trim_motion_text_by_leading_title(sample_row):
    result = trim_motion_text_by_leading_title(sample_row)
    assert result == "a sample text."


def test_trim_whitespace():
    s = pd.Series(["  trim    whitespace    ", "   remove   extra   spaces   "])
    result = trim_whitespace(s)
    expected_result = pd.Series(["trim whitespace", "remove extra spaces"])
    pd.testing.assert_series_equal(result, expected_result)


def test_trim_linebreaks():
    s = pd.Series(["line\nbreaks\r\n", "remove\rline\nbreaks"])
    result = trim_linebreaks(s)
    expected_result = pd.Series(["line breaks", "remove line breaks"])
    pd.testing.assert_series_equal(result, expected_result)


def test_trim_motion_text_by_proposed_decision(sample_row):
    result = trim_motion_text_by_proposed_decision(sample_row)
    assert result == "This is a sample text."


def test_trim_leadning_motivation(sample_row):
    result = trim_leadning_motivation(sample_row)
    assert result == "This is a sample text."


def test_set_empty_when_leading_date():
    row = pd.Series(
        {
            "text": "Stockholm den 123 abc 2023. This is a sample text.",
            "subtitle": "This is",
            "title": "This is",
        }
    )
    result = set_empty_when_leading_date(row)
    assert result == ""


def test_delete_footer(sample_row):
    result = delete_footer(sample_row)
    assert result == "This is a sample text."


def test_prep_text():
    df = pd.DataFrame(
        {
            "text": [
                "Line\nbreaks\r\n",
                "   Trim    whitespace    ",
                "This is a sample text.",
            ],
            "subtitle": [
                "Line",
                "",
                "",
            ],
            "title": [
                "This is",
                "This is",
                "This is",
            ],
        }
    )
    result = prep_text(df, has_title_cols=True)
    expected_result = pd.Series(
        ["breaks", "Trim whitespace", "a sample text."], name="text"
    )
    pd.testing.assert_series_equal(result, expected_result)
