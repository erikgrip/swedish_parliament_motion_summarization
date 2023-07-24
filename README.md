# motion_title_generator

## Table of contents

- [General info](#general-info)
- [Technologies](#technologies)
- [Setup](#setup)
- [Training](#training)
- [Generating Titles](#generating-titles)

## General info

This is a project to

1. fine tune a language model to suggest titles for Swedish parliament motions
2. serve predictions in a simple web app

The project was created with some very concrete inspiration from the course [Full Stack Deep Learning 2021](https://fullstackdeeplearning.com/spring2021/) and from Venelin Valkov's Youtube video [Text summarization with T5 Transformer using PyTorch, PyTorch Lightning and Python](https://www.youtube.com/watch?v=KMyZUIraHio). Go check them out!

__the data__  
Motions from the Swedish Parliament (Riksdagen) can be downloaded via their [open data](https://data.riksdagen.se/in-english/) site. Here is one  [example motion](https://data.riksdagen.se/dokumentstatus/HA022405.html). Use the [API's user interface](https://data.riksdagen.se/dokumentlista/) to read more motions.


## Technologies

The project was created using

- Python 3
- Pytorch and [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
- Docker, Bash, Pytest, Github Actions, Tensorboard and more


## Setup

To run this project you need a system where you can run bash scripts and Linux type commands in your terminal (for example 'echo'). You also need Git, Docker and Conda. See the official docs for instructions on how to install them. With the tools in place, clone the repository using a local terminal. You also need a system where you can run Linux type commands

- [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- [Docker](https://docs.docker.com/get-docker/)
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (only needed to run training)

```
$ git clone https://github.com/erikgrip/swedish_parliament_motion_summarization.git
```

There's some commands to get you started available in the Makefile. If you have Make installed the simply run 'make <command>'. If not, you can still just copy the commands from the Makefile to your terminal.

__make conda-update__  
Sets up a conda environment called _swe-motion-env_ with the contents specified in [environment.yml](environment.yml). Afterwards, make sure to activate the environment.

```bash
conda activate swe-motion-env
```

__make pip-sync__  
Installs the package dependencies in your environment. The dependency specifications can be found in the [requirements/](requirements/) directory.


## Training

When a training is started

1. Training data is downloaded from the Swedish parliament ([read more here](https://data.riksdagen.se/in-english/)), if not already present on your system. It will be kept in the [data/](data/) directory.
2. The data is filtered and preprocessed to get better quality training data.
3. A version of [Google's MT5](https://huggingface.co/docs/transformers/model_doc/mt5) language models will be downloaded fine tuned on the training data using the PyTorch Lightning framework. In the prepped training data each example is a motion text, and it's target the motion's title.
4. There will be a saved model checkpoint for the epoch with the best validation score in the [training/logs/lightning_logs/](training/logs/lightning_logs/) directory.

To list the available training parameters, run:

```bash
make train-help
```

You can use the following command to run a slimmed training loop to make sure everything behaves as intended, but note that a dev run will not produce a model checkpoint.

```bash
make train-dev-run
```

To start a train run, set the desired parameters. For example:

```bash
PYTHONPATH=. python training/run_experiment.py \
        --max_epochs=10 \
        --accelerator='gpu' \
        --devices=1 \
        --early_stopping=3 \
        --model_class=MT5 \
        --data_class=MotionsDataModule
```

See past and running train metrics in tensorboard:

```bash
make tensorboard
```

If you want to convert a checkpoint to a model artifact to use with the web app, use

```bash
PYTHONPATH=. python training/save_checkpoint_to_local_artifact.py --version=<version>
```

where `<version>` is an integer representing a model directory in [training/logs/lightning_logs/](training/logs/lightning_logs/)


## Generating Titles

The project includes a basic web app where you can paste a motion text and get a suggested title. It runs in a docker container and you first need to build a docker image using the script `api_server/build_app_image.sh`.
Make build script executable if needed.

```bash
chmod +x api_server/build_app_image.sh
```

The script takes two command line arguments:

- model_type - Specifiy where the model is located. Either hf (for huggingface) or local.
- model_path - Either a huggingface repo (`user-name/repo-name`) or a local path (for example `motion_title_generator/artifacts/version1_epoch001_val_loss0.01`).

If you just want to get a sample app going you can use:

```bash
make build-sample-app-image
```

When the Flask server is running it outputs a link that you can follow to get to the app.
