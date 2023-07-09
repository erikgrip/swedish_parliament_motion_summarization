from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from motion_title_generator.data.base_data_module import BaseDataModule
from utils.log import logger

DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded/zipped"


def download_motion_zip_files(
    dl_dirname: Path = DOWNLOADED_DATA_DIRNAME, file_type="json"
):
    """Download a collection of zipped directories to the dl_dirname
    directory. Returns a list of files downloaded.

    Args:
    ----
    - file_type     Selects the type of files that the directories
                    contain. See https://data.riksdagen.se/data/dokument/
                    for available types

    Example:
    -------
    downloaded_files = download_motion_zip_files('csv')
    """
    dl_dirname.mkdir(parents=True, exist_ok=True)

    base_url = "https://data.riksdagen.se/"
    doc_catalogue_url = base_url + "dataset/katalog/dataset.xml"
    response = requests.get(doc_catalogue_url, allow_redirects=True, timeout=10.0)
    soup = BeautifulSoup(response.content, features="html.parser")
    doc_list = soup.datasetlista.findAll("dataset") if soup.datasetlista else []
    logger.info("Dowloading files from %s.", base_url)
    paths_to_downloads = []
    for doc in tqdm(doc_list):
        if (doc.typ.string == "mot") & (doc.format.string == file_type):
            output_file_path = Path(dl_dirname / doc.filnamn.string)
            if not output_file_path.is_file():
                zip_arch_url = base_url + doc.url.string
                response = requests.get(
                    zip_arch_url, allow_redirects=True, timeout=10.0
                )
                with open(output_file_path, "wb") as f:
                    logger.debug(
                        "Downloading raw dataset from %s to %s ...",
                        zip_arch_url,
                        output_file_path,
                    )
                    f.write(response.content)
                paths_to_downloads.append(output_file_path)
    return paths_to_downloads


if __name__ == "__main__":
    downloaded = download_motion_zip_files(
        dl_dirname=DOWNLOADED_DATA_DIRNAME, file_type="json"
    )
    print("Downloaded zip files:", [str(p) for p in downloaded])
