# pylint: disable=missing-function-docstring
import zipfile

import pytest

from training_data_pipeline.utils.zip_data_reader import (
    parse_html_text,
    read_motions_from_zip_arch,
)


@pytest.fixture(name="sample_zip_file")
def fixture_sample_zip_file(tmp_path):
    # Create a sample zip file for testing
    zip_file = tmp_path / "sample.zip"
    with zipfile.ZipFile(zip_file, "w") as f:
        f.writestr(
            "motion1.json",
            '{"dokumentstatus": {"dokument": {'
            + '"titel": "Motion 1", '
            + '"dok_id": "1", '
            + '"datum": "2023-01-01", '
            + '"systemdatum": "2023-01-01", '
            + '"subtitel": "Subtitle 1", '
            + '"html": "<html><body>Sample HTML for motion 1</body></html>"}, '
            + '"dokintressent": {"intressent": '
            + '{"partibet": "A", "namn": "Name 1"}}}}',
        )
        f.writestr(
            "motion2.json",
            '{"dokumentstatus": {"dokument": {'
            + '"titel": "Motion 2", '
            + '"dok_id": "2", '
            + '"datum": "2023-01-02", '
            + '"systemdatum": "2023-01-03", '
            + '"subtitel": "Subtitle 2", '
            + '"html": "<html><body>Sample HTML for motion 2</body></html>"}, '
            + '"dokintressent": {"intressent": ['
            + '{"partibet": "B", "namn": "Name 2"}, '
            + '{"partibet": "C", "namn": "Name 3"}]}}}',
        )
        f.writestr(
            "motion3.json",
            '{"dokumentstatus": {"dokument": {'
            + '"titel": "Motionen utgår", '  # Should be skipped
            + '"dok_id": "3", '
            + '"datum": "2023-01-03", '
            + '"systemdatum": "2023-01-03", '
            + '"subtitel": "Subtitle 3", '
            + '"html": "<html><body>Sample HTML for motion 3</body></html>"}, '
            + '"dokintressent": {"intressent": {"partibet": "A", "namn": "Name 1"}}}}',
        )
    yield zip_file


def test_read_motions_from_zip_arch(sample_zip_file):
    # Test reading motions from a sample zip file
    data = read_motions_from_zip_arch(sample_zip_file)
    assert len(data) == 2
    assert data[0]["id"] == "1"
    assert data[0]["title"] == "Motion 1"
    assert data[0]["date"] == "2023-01-01"
    assert data[0]["file_date"] == "2023-01-01"
    assert data[0]["subtitle"] == "Subtitle 1"
    assert data[0]["text"] == "Sample HTML for motion 1"
    assert data[1]["id"] == "2"
    assert data[1]["title"] == "Motion 2"
    assert data[1]["date"] == "2023-01-02"
    assert data[1]["file_date"] == "2023-01-03"
    assert data[1]["subtitle"] == "Subtitle 2"
    assert data[1]["text"] == "Sample HTML for motion 2"


def test_parse_html_text():
    # Test parsing HTML text
    html = "<html><body>Sample HTML</body></html>"
    expected_text = "Sample HTML"
    assert parse_html_text(html) == expected_text
