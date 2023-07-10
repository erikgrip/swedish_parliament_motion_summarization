# pylint: disable=missing-function-docstring
from pathlib import Path

import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype
from pandas.testing import assert_frame_equal

from training_data_pipeline.utils.preprocessor import (
    filter_nan_rows,
    filter_short_motions,
    filter_titles,
    load_dataframe,
)


@pytest.fixture(name="sample_dataframe")
def fixture_sample_dataframe():
    """Create a sample dataframe for testing."""
    data = {
        "date": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
        "file_date": ["2022-01-05", "2022-01-06", "2022-01-07", "2022-01-08"],
        "title": ["Title 1", "Title 2", "Title 3", "Title 4"],
        "text": [
            "Text 1",
            "Text 2",
            "Text 3",
            "Text 4",
        ],
    }
    return pd.DataFrame(data)


def test_load_dataframe(sample_dataframe):
    # Create a temporary file for testing
    temp_file = "test_data.pkl"
    sample_dataframe.to_pickle(temp_file)

    loaded_df = load_dataframe(data_path=Path(temp_file))
    assert_frame_equal(
        loaded_df[["title", "text"]], sample_dataframe[["title", "text"]]
    )
    assert is_datetime64_any_dtype(loaded_df["date"])
    assert is_datetime64_any_dtype(loaded_df["file_date"])

    # Clean up temporary file
    Path(temp_file).unlink()


def test_filter_nan_rows(sample_dataframe):
    sample_dataframe.loc[0, "title"] = pd.NA
    filtered_df = filter_nan_rows(sample_dataframe)

    assert len(filtered_df) == 3
    assert filtered_df["text"].values.tolist() == ["Text 2", "Text 3", "Text 4"]


def test_filter_short_motions(sample_dataframe):
    sample_dataframe.loc[3, "text"] = 20 * "bla, bla, bla,"
    filtered_df = filter_short_motions(sample_dataframe)

    assert len(filtered_df) == 1
    assert filtered_df["title"].values.tolist() == ["Title 4"]


def test_filter_titles(sample_dataframe):
    # Modify the titles to include a generic starting phrase
    sample_dataframe["title"] = [
        "Med anledning av prop ABC",
        "Title 2",
        "Title 3",
        "med anledning av prop XYZ",
    ]
    filtered_df = filter_titles(sample_dataframe)

    assert len(filtered_df) == 2
    assert filtered_df["title"].values.tolist() == ["Title 2", "Title 3"]
