from pathlib import Path

import pandas as pd

from utils.log import logger
from utils.text import prep_text, trim_whitespace

DOWNLOADED_DATA_DIRNAME = Path(__file__).resolve().parents[3] / "data" / "downloaded"
INPUT_DATA_PATH = DOWNLOADED_DATA_DIRNAME / "raw_swe_parl_mot.pkl"
OUTPUT_DATA_PATH = DOWNLOADED_DATA_DIRNAME / "prepped_training_data.feather"


def load_dataframe(data_path=INPUT_DATA_PATH) -> pd.DataFrame:
    """Load data from pickle file into a pandas dataframe."""
    logger.info("Loading data from %s into pandas dataframe.", data_path)
    try:
        df = pd.DataFrame(pd.read_pickle(data_path))  # nosec
    except FileNotFoundError:
        logger.error("No data found at %s.", data_path)
        raise
    df["date"] = pd.to_datetime(df["date"])
    df["file_date"] = pd.to_datetime(df["file_date"])
    return df


def filter_nan_rows(df) -> pd.DataFrame:
    """Filter out rows with missing values."""
    pre_filter_len = len(df)
    df = df.dropna()
    logger.info("Filtered %s rows with missing values.", (pre_filter_len - len(df)))
    return df


def filter_short_motions(df) -> pd.DataFrame:
    """Filter out rows with motions shorter than 150 characters."""
    pre_filter_len = len(df)
    df = df.loc[df["text"].str.len() >= 150].reset_index(drop=True)
    logger.info(
        "Filtered %s texts shorter than 150 characters.", (pre_filter_len - len(df))
    )
    return df


def filter_titles(df) -> pd.DataFrame:
    """Filter out rows with generic titles that won't make good training examples."""
    pre_filter_len = len(df)
    df = df.loc[
        ~df["title"].str.lower().str.startswith("med anledning av prop")
    ].reset_index(drop=True)
    logger.info("Filtered %s texts based on their title.", (pre_filter_len - len(df)))
    return df


# pylint: disable=unsupported-assignment-operation,unsubscriptable-object
def prep_training_dataset():
    """Pipeline to format and filter data."""
    df = load_dataframe()
    logger.info("Preprocessing data ...")

    df = filter_nan_rows(df)

    # Prep target and feature texts
    df["title"] = trim_whitespace(df["title"])
    df["text"] = prep_text(df)

    df = filter_short_motions(df)
    df = filter_titles(df)

    logger.info("Number of rows remaining: %s", len(df))
    df.to_feather(path=OUTPUT_DATA_PATH)
    logger.info("Preprocessed data saved to %s", OUTPUT_DATA_PATH)


if __name__ == "__main__":
    prep_training_dataset()
