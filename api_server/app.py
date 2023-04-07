import os
import logging

from flask import Flask, request, render_template
import pandas as pd

from text_summarizer.motion_title_generator import MotionTitleGenerator
from training_dataset_downloader.src.training_dataset_preprocessor import prep_text

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU

INDEX_TEXT = "App is running. Make predictions at http://localhost:8000/v1/predict"


app = Flask(__name__)
model = MotionTitleGenerator()
logging.basicConfig(level=logging.INFO)


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
    """Get text from webb app and render prediction."""
    text = request.form["text"]
    # Clean text in the same way as in training.
    text = prep_text(pd.DataFrame({"text": [text]}), has_title_cols=False)[0]
    if len(text) < 300:
        pred = "Please enter a longer text"
    else:
        pred = model.predict(text)
        logging.info("pred %s", pred)
    return render_template("predict_form.html", text=text, pred=pred)


def main():
    """Run the app."""
    app.run(host="0.0.0.0", port=8000, debug=False)  # nosec


if __name__ == "__main__":
    main()
