import pandas as pd

from motion_title_generator.data.base_data_module import BaseDataModule
from utils.log import logger
from utils.text import prep_text, trim_whitespace

DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded"
INPUT_DATA_PATH = DOWNLOADED_DATA_DIRNAME / "raw_swe_parl_mot.pkl"
OUTPUT_DATA_PATH = DOWNLOADED_DATA_DIRNAME / "prepped_training_data.feather"


def load_dataframe(data_path=INPUT_DATA_PATH):
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


def prep_training_dataset():
    """Pipeline to format and filter data."""
    df = load_dataframe()
    logger.info("Preprocessing data ...")

    pre_filter_len = len(df)
    df = df.dropna()
    logger.info("Filtered %s rows with missing values.", (pre_filter_len - len(df)))

    # Prep target and feature texts
    df["title"] = trim_whitespace(df["title"])
    df["text"] = prep_text(df)

    pre_filter_len = len(df)
    df = df.loc[df["text"].str.len() >= 150].reset_index(drop=True)
    logger.info(
        "Filtered %s texts shorter than 150 characters.", (pre_filter_len - len(df))
    )

    pre_filter_len = len(df)
    df = df.loc[
        ~df["title"].str.lower().str.startswith("med anledning av prop")
    ].reset_index(drop=True)
    logger.info("Filtered %s texts with generic title.", (pre_filter_len - len(df)))

    logger.info("Number of rows remaining: %s", len(df))
    df.to_feather(path=OUTPUT_DATA_PATH)
    logger.info("Preprocessed data saved to %s", OUTPUT_DATA_PATH)


if __name__ == "__main__":
    prep_training_dataset()
