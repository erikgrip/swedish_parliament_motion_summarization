from pathlib import Path
from typing import Dict
import argparse
import json
import logging
import re
import zipfile

from bs4 import BeautifulSoup
from spacy.lang.sv import Swedish
from torch.utils.data import random_split, DataLoader
import pandas as pd
import requests

from text_summarizer.data.base_data_module import (
    BaseDataModule,
    load_and_print_info)
from text_summarizer.data.util import (
    load_pickle,
    save_pickle,
    trim_whitespace,
    text_to_lower)
from text_summarizer import util


DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded"


def _download_raw_dataset(metadata: Dict, dl_dirname: Path) -> Path:
    dl_dirname.mkdir(parents=True, exist_ok=True)
    filename = dl_dirname / metadata["filename"]
    if filename.exists():
        return filename
    print(f"Downloading raw dataset from {metadata['url']} to {filename}...")
    util.download_url(metadata["url"], filename)
    print("Computing SHA-256...")
    sha256 = util.compute_sha256(filename)
    if sha256 != metadata["sha256"]:
        raise ValueError("Downloaded data file SHA-256 does not match that listed in metadata document.")
    return filename

def _download_motion_zip_files(dl_dirname: Path, file_type='html'):
    '''Downloads a collection of zipped directories to the dl_dirname
    directory. Returns a list of all files in the download directory.

    Args:
    - file_type     Selects the type of files that the directories
                    contain. See https://data.riksdagen.se/data/dokument/
                    for available types

    Example:
    downloaded_files = download_motion_zip_files('csv')
    '''
    dl_dirname.mkdir(parents=True, exist_ok=True)

    base_url = 'https://data.riksdagen.se/'
    doc_catalogue_url = base_url + 'dataset/katalog/dataset.xml'
    response = requests.get(doc_catalogue_url, allow_redirects=True)
    soup = BeautifulSoup(response.content, features="html.parser")
    doc_list = soup.datasetlista.findAll('dataset')

    for doc in doc_list:
        if (doc.typ.string == 'mot') & (doc.format.string == file_type):
            output_file_path = Path(dl_dirname / doc.filnamn.string)
            if not output_file_path.is_file():
                zip_arch_url = base_url + doc.url.string
                response = requests.get(zip_arch_url, allow_redirects=True)
                with open(output_file_path, 'wb') as f:
                    print(f"Downloading raw dataset from {zip_arch_url} to {output_file_path}...")
                    f.write(response.content)
    return dl_dirname.glob('*.zip')
    

def _read_motions_from_zip_arch(zip_arch):
    '''Read motion files from local zipped directories and return
    a list of dictionaries with one entry per motion. Each dict hold info
    about the document's ID, date, title and text.

    Args:
    zip_arg -   The file path of the zipped directory

    Example:
    import pandas as pd
    d = read_motions_from_zip_arch('data/raw/mot-2018-2021.json.zip')
    df = pd.DataFrame(d)
    '''
    docs = []
    with zipfile.ZipFile(zip_arch) as z:
        for filename in z.namelist():
            with z.open(filename) as f:
                data = f.read()
                d = json.loads(data.decode("utf-8-sig"))
                document = d['dokumentstatus']['dokument']

                doc = {}
                try:
                    doc['id'] = document['dok_id']
                    doc['date'] = document['datum']
                    doc['title'] = document['titel']
                    doc['subtitle'] = document['subtitel']
                    doc['text'] = _parse_html_text(document['html'])
                    authors =  d['dokumentstatus']['dokintressent']['intressent']
                    if isinstance(authors, list):
                        doc['main_author'] = authors[0]['namn']
                        doc['author_party'] = authors[0]['partibet']
                    else:
                        doc['main_author'] = authors['namn']
                        doc['author_party'] = authors['partibet']
                    docs.append(doc)
                except KeyError as e:
                    logging.info('Did not find key %s in motion id=%s, title=%s',
                                 e, document['dok_id'], document['titel'])
                    pass
                except TypeError as e:
                    logging.error('Failed to read motion: %s', e)
                    pass
    return docs


def _parse_html_text(html: str):
    soup = BeautifulSoup(html, features="html.parser")

    # Drop script and style elements
    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    return text


class SweParliamentMotions(BaseDataModule):
    def __init__(self, args: argparse.Namespace) -> None:
        self.data_dir = DOWNLOADED_DATA_DIRNAME
    
    def prepare_data(self):
        # Define steps that should be done
        # on only one GPU, like getting data.
        download_motion_zip_files(self.data_dir + '/zipped', 'html')
    
    def setup(self, stage=None):
        # Define steps that should be done on 
        # every GPU, like splitting data, applying
        # transform etc.
        pass
    
    def train_dataloader(self):
        # Return DataLoader for Training Data here
        pass
    
    def val_dataloader(self):
        # Return DataLoader for Validation Data here
        pass
    
    def test_dataloader(self):
        # Return DataLoader for Testing Data here
        pass


if __name__ == "__main__":
    load_and_print_info(SweParliamentMotions)