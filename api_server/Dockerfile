# Re-declare inside each build stage to use in that stage
ARG MODEL_PATH
ARG MODEL_TYPE

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


# 2. Extend image to include local model artifacts
# ------------------------------------------------
FROM build_hf AS build_local

ARG MODEL_PATH

COPY $MODEL_PATH ${MODEL_PATH}


# 3. Select base image using MODEL_TYPE arg
# -----------------------------------------
FROM build_${MODEL_TYPE} AS build

ARG MODEL_PATH

# Run the web server
EXPOSE 8000
ENV PYTHONPATH /repo
ENV HF_REPO_OR_ARTIFACT_PATH ${MODEL_PATH}
CMD python3 /repo/api/app.py