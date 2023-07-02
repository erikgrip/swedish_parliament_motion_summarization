import os
import unittest

from api_server.app import app

os.environ["CUDA_VISIBLE_DEVICES"] = ""

INVALID_TEXT = "This one's too short."
VALID_TEXT = """Persontrafik på järnvägen genom Sörmland på den så kallade \
TGOJ-banan (Trafikaktie­bolaget Grängesberg–Oxelösunds Järnvägar) bör \
återinföras. Järnvägen sträcker sig från Oxelösund, via Flen och Eskilstuna \
vidare norrut. I dagsläget används sträckanmellan Oxelösund och Flen för \
godstrafik men inte persontrafik."""


class TestIntegrations(unittest.TestCase):
    """Unittests for the web app."""

    def setUp(self):
        self.app = app.test_client()

    def test_index(self):
        """Test that the health check page returns the expected text."""
        response = self.app.get("/health")
        assert response.get_data().decode() == "OK"

    def test_predict_valid_input(self):
        """Test that the predict page contains the expected text for valid input."""
        response = self.app.post("/predict", data={"text": VALID_TEXT})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Suggested title:", response.data)
        self.assertNotIn(b"Please enter a longer text", response.data)

    def test_predict_invalid_input(self):
        """Test that the predict page contains the expected text for too short input."""
        response = self.app.post("/predict", data={"text": INVALID_TEXT})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Please enter a longer text", response.data)


if __name__ == "__main__":
    unittest.main()
