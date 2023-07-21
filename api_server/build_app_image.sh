#!/bin/bash

# Validate the number of arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <model_type> <model_path> "
  echo "model_type: one of 'hf' or 'local'"
  echo "model_path: path to the model"
  echo "Example huggingface: $0 hf erikgrip2/mt5-finetuned-for-motion-title"
  echo "Example local: $0 local motion_title_generator/artifacts/version1_epoch001_val_loss0.1"
  exit 1
fi

# Get the source directory and image name from the command-line arguments
MODEL_TYPE="$1"
HF_REPO_OR_ARTIFACT_PATH="$2"

docker build \
    -t "motion_title_app:latest" \
    -f api_server/Dockerfile \
    --build-arg MODEL_TYPE="$MODEL_TYPE" \
    --build-arg MODEL_PATH="$HF_REPO_OR_ARTIFACT_PATH" \
    .


