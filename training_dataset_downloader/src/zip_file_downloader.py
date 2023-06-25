from pathlib import Path

from bs4 import BeautifulSoup
import requests

from motion_title_generator.data.base_data_module import BaseDataModule


DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded/zipped"


def download_motion_zip_files(
    dl_dirname: Path = DOWNLOADED_DATA_DIRNAME, file_type="json"
):
    """Downloads a collection of zipped directories to the dl_dirname
    directory. Returns a list of files downloaded.

    Args:
    - file_type     Selects the type of files that the directories
                    contain. See https://data.riksdagen.se/data/dokument/
                    for available types

    Example:
    downloaded_files = download_motion_zip_files('csv')
    """
    dl_dirname.mkdir(parents=True, exist_ok=True)

    base_url = "https://data.riksdagen.se/"
    doc_catalogue_url = base_url + "dataset/katalog/dataset.xml"
    response = requests.get(doc_catalogue_url, allow_redirects=True)
    soup = BeautifulSoup(response.content, features="html.parser")
    doc_list = soup.datasetlista.findAll("dataset") if soup.datasetlista else []

    downloaded = []
    for doc in doc_list:
        if (doc.typ.string == "mot") & (doc.format.string == file_type):
            output_file_path = Path(dl_dirname / doc.filnamn.string)
            if not output_file_path.is_file():
                zip_arch_url = base_url + doc.url.string
                response = requests.get(zip_arch_url, allow_redirects=True)
                with open(output_file_path, "wb") as f:
                    print(
                        f"Downloading raw dataset from {zip_arch_url} to {output_file_path}..."
                    )
                    f.write(response.content)
                downloaded.append(output_file_path)
    return downloaded


if __name__ == "__main__":
    downloaded = download_motion_zip_files(
        dl_dirname=DOWNLOADED_DATA_DIRNAME, file_type="json"
    )
    print("Downloaded zip files:", [str(p) for p in downloaded])
