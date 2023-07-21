ARG MODEL_TYPE
ARG MODEL_PATH

# 1. Use this image with model stored at huggingface
# --------------------------------------------------
FROM python:3.9-buster AS build_hf

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Install Python dependencies
COPY requirements/prod.txt ./requirements.txt
RUN sed -i 's/cu113/cpu/g' requirements.txt
RUN pip install --upgrade pip~=21.0.0
RUN pip install -r requirements.txt

COPY api_server/ ./api
COPY motion_title_generator/motion_title_generator.py ./motion_title_generator/motion_title_generator.py
COPY utils/ ./utils


# 2. Use this image to include local model artifacts
# --------------------------------------------------
FROM build_hf AS build_local
COPY $MODEL_PATH ./motion_title_generator/artifacts


# 3. Use relevant image version
# -----------------------------
FROM build_${MODEL_TYPE} AS build

# Run the web server
EXPOSE 8000
ENV PYTHONPATH /repo
CMD python3 /repo/api/app.py