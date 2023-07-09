from unittest.mock import call, patch

import pytest

from training_data_pipeline.utils.file_downloader import download_motion_zip_files


@pytest.fixture(name="mock_requests_get")
def fixture_mock_requests_get():
    """Mock the requests.get function."""
    with patch("training_data_pipeline.utils.file_downloader.requests.get") as mock_get:
        yield mock_get


def test_download_motion_zip_files(mock_requests_get, tmp_path):
    """Test the calls to requests.get and the return value of the function."""
    # Mock the requests.get function and its return value
    mock_response = mock_requests_get.return_value
    mock_response.content = b"<datasetlista><dataset><typ>mot</typ><format>json</format><filnamn>file1.json</filnamn><url>dataset/file1.zip</url></dataset></datasetlista>"  # noqa: E501

    # Call the function under test
    dl_dirname = tmp_path / "downloads"
    downloaded_files = download_motion_zip_files(
        dl_dirname=dl_dirname, file_type="json"
    )

    # Assertions
    mock_requests_get.assert_has_calls(
        [
            call(
                "https://data.riksdagen.se/dataset/katalog/dataset.xml",
                allow_redirects=True,
                timeout=10.0,
            ),
            call(
                "https://data.riksdagen.se/dataset/file1.zip",
                allow_redirects=True,
                timeout=10.0,
            ),
        ],
        any_order=True,
    )
    assert len(downloaded_files) == 1
    assert downloaded_files[0] == dl_dirname / "file1.json"
