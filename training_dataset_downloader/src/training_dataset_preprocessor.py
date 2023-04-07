import pickle
import re

import pandas as pd

from text_summarizer.data.base_data_module import BaseDataModule
from training_dataset_downloader.src.zip_data_reader import OUTPUT_PATH


DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded"
INPUT_DATA_PATH = DOWNLOADED_DATA_DIRNAME / "raw_swe_parl_mot.pkl"
OUTPUT_DATA_PATH = DOWNLOADED_DATA_DIRNAME / "prepped_training_data.feather"


def _trim_motion_text_by_subtitle(row):
    try:
        split = row["text"].split(row["subtitle"], 1)
        return split[-1].strip()
    except Exception as e:
        print(e)
        return row["text"]


def _trim_motion_text_by_leading_title(row):
    if row["text"].startswith(row["title"]):
        text = row["text"].split(row["title"], 1)[-1].strip()
    else:
        text = row["text"]
    return text


def _trim_whitespace(s):
    """Remove trailing and multiple whitespaces"""
    return s.replace("\s+", " ", regex=True).str.strip()


def _trim_linebreaks(s):
    return s.replace("\n", " ").replace("\r", "").str.strip()


def _trim_motion_text_by_proposed_decision(row):
    # .+?(?=(\. [A-ZÅÄÖ])) --> All up to first '.' followed by whitespace and
    # upper case letter. Not watertight by any means but a reasonable best effort.
    # TODO: the original .json holds the proposed decisions by document under
    # the key 'dokforslag'. Use that instead.
    split = re.split(
        "Förslag till riksdagsbeslut .+?\. (?=([A-ZÅÄÖ]))|"
        + "Riksdagen tillkännager för [A-Öa-ö]+ som sin mening .+?\. (?=([A-ZÅÄÖ]))|"
        + "Riksdagen bemyndigar .+?\. (?=([A-ZÅÄÖ]))|"
        + "Riksdagen beslutar om .+?\. (?=([A-ZÅÄÖ]))|"
        + "Härmed hemställs att riksdagen .+?\. (?=([A-ZÅÄÖ]))|"
        + "Med hänvisning till vad so(?:m|rn) anförts .+?\. (?=([A-ZÅÄÖ]))|"
        + "Riksdagen ställer sig bakom det som anförs .+?\. (?=([A-ZÅÄÖ]))",
        row["text"],
    )
    return split[-1].strip()


def _trim_leadning_motivation(row):
    if re.match(r"^Motivering [A-ZÅÄÖ\d]", row["text"]):
        row["text"] = row["text"].split("Motivering", 1)[-1].strip()
    return row["text"]


def _set_empty_when_leadning_date(row):
    if re.match(r"^Stockholm den [\d]+ [a-z]+ \d{4}", row["text"]):
        row["text"] = ""
    return row["text"]


def _delete_footer(row):
    return re.sub(r"(?<=\.) Stockholm den [\d]+ [a-z]+ \d{4} .+", "", row["text"])


def prep_text(df, has_title_cols=True):
    """Pipeline to preproces text column."""
    df["text"] = _trim_linebreaks(df["text"])
    df["text"] = _trim_whitespace(df["text"])
    if has_title_cols:
        df["text"] = df.apply(_trim_motion_text_by_subtitle, axis=1)
        df["text"] = df.apply(_trim_motion_text_by_leading_title, axis=1)
    df["text"] = df.apply(_trim_motion_text_by_proposed_decision, axis=1)
    df["text"] = df.apply(_trim_leadning_motivation, axis=1)
    df["text"] = df.apply(_set_empty_when_leadning_date, axis=1)
    df["text"] = df.apply(_delete_footer, axis=1)
    return df["text"]


def prep_training_dataset(data_path=INPUT_DATA_PATH):
    """Pipeline to format and filter data."""
    with open(data_path, "rb") as f:
        print("Preprocessing data ...")
        df = pd.DataFrame(pickle.load(f))

    pre_filter_len = len(df)
    df = df.dropna()
    print(f"Filtered {pre_filter_len - len(df)} rows with missing values.")

    df["date"] = pd.to_datetime(df["date"])
    df["file_date"] = pd.to_datetime(df["file_date"])

    # Prep target and feature texts
    df["title"] = _trim_whitespace(df["title"])
    df["text"] = prep_text(df)

    pre_filter_len = len(df)
    df = df.loc[df["text"].str.len() >= 150].reset_index(drop=True)
    print(f"Filtered {pre_filter_len - len(df)} texts shorter than 150 characters.")
    pre_filter_len = len(df)
    df = df.loc[
        ~df["title"].str.lower().str.startswith("med anledning av prop")
    ].reset_index(drop=True)
    print(f"Filtered {pre_filter_len - len(df)} texts with generic title.")
    print(f"Number of rows remaining: {len(df)}")

    df.to_feather(path=OUTPUT_DATA_PATH)
    print("Preprocessed data saved to", OUTPUT_DATA_PATH)


if __name__ == "__main__":
    prep_training_dataset()
