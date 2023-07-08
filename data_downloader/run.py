from .src.zip_file_downloader import download_motion_zip_files
from .src.zip_data_reader import read_zip_file_data_to_pkl
from .src.training_dataset_preprocessor import prep_training_dataset


def main():
    """Download and prep motions data for training."""
    downloaded = download_motion_zip_files()
    if len(downloaded) >= 1:
        read_zip_file_data_to_pkl()
    prep_training_dataset()
