import logging
import os

from flask import Flask, render_template, request
from pandas import DataFrame

from motion_title_generator.motion_title_generator import MotionTitleGenerator
from utils.text import prep_text

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU

INDEX_TEXT = "App is running. Make predictions at http://localhost:8000/v1/predict"

app = Flask(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
model = MotionTitleGenerator()

@app.route("/")
def index():
    """Provide simple health check route."""
    return INDEX_TEXT


@app.route("/v1/predict")
def predict_form():
    """Render prediction input form."""
    return render_template("predict_form.html")


@app.route("/v1/predict", methods=["GET", "POST"])
def predict():
    """Get text from web app and render prediction."""
    text = request.form.get("text")
    if not text:
        return "Please enter some text"

    text = prep_text(DataFrame({"text": [text]}), has_title_cols=False)[0]
    if len(text) < 300:
        pred = "Please enter a longer text"
    else:
        pred = model.predict(text)
        logging.info("pred %s", pred)

    return render_template("predict_form.html", text=text or "", pred=pred or "")


def main():
    """Run the app."""
    app.run(host="0.0.0.0", port=8000, debug=False)  # nosec


if __name__ == "__main__":
    main()
