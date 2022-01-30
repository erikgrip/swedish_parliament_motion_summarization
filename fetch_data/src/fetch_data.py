import requests
import json
from bs4 import BeautifulSoup
import zipfile
import os
import re
import pickle
import time
import pandas as pd
import pathlib
import logging


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap


def download_motion_zip_files(file_type='html'):
    '''Downloads a collection of zipped directories to the /data/raw
    directory. Returns a list of the downloaded files' names

    Args:
    - file_type     Selects the type of files that the directories
                    contain. See https://data.riksdagen.se/data/dokument/
                    for available types

    Example:
    downloaded_files = download_motion_zip_files('csv')
    '''
    base_url = 'https://data.riksdagen.se/'
    doc_catalogue_url = base_url + 'dataset/katalog/dataset.xml'

    response = requests.get(doc_catalogue_url, allow_redirects=True)
    soup = BeautifulSoup(response.content, features="html.parser")
    doc_list = soup.datasetlista.findAll('dataset')

    # Create output dir if not exists
    output_dir = 'data/raw'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    downloaded_archive_names = []
    for doc in doc_list:
        if (doc.typ.string == 'mot') & (doc.format.string == file_type):
            output_file_path = output_dir + '/' + doc.filnamn.string
            zip_arch_url = base_url + doc.url.string
            response = requests.get(zip_arch_url, allow_redirects=True)
            with open(output_file_path, 'wb') as f:
                f.write(response.content)
            downloaded_archive_names.append(output_file_path)
    return downloaded_archive_names


@timing
def _trim_whitespace(text):
    return re.sub('\s+', ' ', text).strip()


@timing
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


def read_motions_from_zip_arch(zip_arch):
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


def save_pickle(obj, path):
    target_file = open(path, "wb")
    pickle.dump(obj, target_file)
    target_file.close()


def read_pickle(path):
    f = open(path, "rb")
    return pickle.load(f)


def main():
    # Create output dir if not exists
    output_dir = 'logs'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename='logs/fetch.log',
                        encoding='utf-8',
                        level=logging.DEBUG)

    dl_zip_file_paths = download_motion_zip_files('json')

    mot_dict = []
    for z in dl_zip_file_paths:
        mot_dict += read_motions_from_zip_arch(z)

    save_pickle(mot_dict, 'data/docs_dict.pkl')
