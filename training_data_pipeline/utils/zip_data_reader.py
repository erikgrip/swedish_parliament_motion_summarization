import json
import pickle
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup

from utils.log import logger

DOWNLOADED_DATA_DIRNAME = Path(__file__).resolve().parents[2] / "data" / "downloaded"
ZIP_DIR = DOWNLOADED_DATA_DIRNAME / "zipped"
OUTPUT_PATH = DOWNLOADED_DATA_DIRNAME / "raw_swe_parl_mot.pkl"


def read_zip_file_data_to_pkl(zip_dir=ZIP_DIR, output_path=OUTPUT_PATH):
    """Read data from directory with zip files and save it to a pickle file."""
    logger.info("Reading data from zip files in %s ...", zip_dir)
    data = []
    for zip_file in zip_dir.glob("*.zip"):
        data_list = read_motions_from_zip_arch(zip_file)
        data.extend(data_list)

    logger.info("Saving data ...")
    with open(output_path, "wb") as target_file:
        pickle.dump(data, target_file)
    logger.info("The %s file was saved at %s", target_file, output_path)


def read_motions_from_zip_arch(zip_arch):
    """Read motion files from local zipped directories and return
    a list of dictionaries with one entry per motion. Each dict hold info
    about the document's ID, date, title and text.

    Args:
    ----
    zip_arg -   The file path of the zipped directory

    Example:
    -------
    import pandas as pd
    data = read_motions_from_zip_arch('data/raw/mot-2018-2021.json.zip')
    df = pd.DataFrame(data)
    """
    parsed_docs = []
    total_docs = 0
    with zipfile.ZipFile(zip_arch) as zipped:
        for filename in zipped.namelist():
            total_docs += 1
            with zipped.open(filename) as f:
                data = f.read()
                data = json.loads(data.decode("utf-8-sig"))
                try:
                    document = data["dokumentstatus"]["dokument"]
                except TypeError as e:
                    logger.debug("Failed to read motion %s: %s", filename, e)
                    continue

                doc = {}
                try:
                    doc["title"] = document["titel"]
                    if doc["title"] == "Motionen utgår":
                        continue

                    doc["id"] = document["dok_id"]
                    doc["date"] = document["datum"]
                    doc["file_date"] = document["systemdatum"]
                    doc["subtitle"] = document["subtitel"]
                    doc["text"] = parse_html_text(document["html"])
                    authors = data["dokumentstatus"]["dokintressent"]["intressent"]
                    if isinstance(authors, list):
                        doc["main_author"] = authors[0]["namn"]
                        doc["author_party"] = authors[0]["partibet"]
                    else:
                        doc["main_author"] = authors["namn"]
                        doc["author_party"] = authors["partibet"]
                    parsed_docs.append(doc)
                except (KeyError, TypeError) as e:
                    logger.debug(
                        "Did not find key %s in motion id=%s, title=%s",
                        e,
                        document["dok_id"],
                        document["titel"],
                    )
    logger.info(
        "Successfully parsed %s of %s documents from %s.",
        len(parsed_docs),
        total_docs,
        zip_arch,
    )
    return parsed_docs


def parse_html_text(html: str) -> str:
    """Parse html text and return a string with the text content."""
    soup = BeautifulSoup(html, features="html.parser")

    # Drop script and style elements
    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = " ".join(chunk for chunk in chunks if chunk)
    return text
