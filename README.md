# swedish_parliament_motion_summarization

Start tensorboard:

```bash
tensorboard --logdir training/logs/lightning_logs
```

Push checkpoint to Huggingface Hub:

```bash
PYTHONPATH=. python motion_title_generator/save_checkpoint_to_huggingface.py --version=2 --hf_model="erikgrip2/mt5-finetuned-for-motion-title"  --hf_user="erikgrip2"
```

Build app Docker image:

```bash
# Navigate to project's root directory and run
docker build -t motion_title_app -f api_server/Dockerfile .
```

Run app in Docker container:

```bash
docker run -p 8000:8000 --name app_container motion_title_app
```

Run app without docker:

```bash
PYTHONPATH=. python api_server/app.py
```

Make build script executable if needed:

```bash
chmod +x api_server/build_app_image.sh
```

# motion_title_generator

## Table of contents

- [General info](#general-info)
- [Technologies](#technologies)
- [Setup](#setup)
- [Training](#training)
- [Output data](#output-data)

## General info

This is a project to

1. fine tune a language model to suggest titles for Swedish parliament motions
2. serve predictions in a simple web app

## Technologies

Project is created using

- Python 3
- Pytorch and [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
- Docker, Bash, Pytest, Github Actions, Tensorboard and more

## Setup

To run this project you need a system where you can run bash scripts and Linux type commands in your terminal (for example 'echo'). You also need Git, Docker and Conda. See the official docs for instructions on how to install them. With the tools in place, clone the repository using a local terminal. You also need a system where you can run Linux type commands 
* [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
* [Docker](https://docs.docker.com/get-docker/)
* [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (only needed to run training)

```
$ git clone https://github.com/erikgrip/swedish_parliament_motion_summarization.git
```

There's some commands to get you started available in the Makefile. If you have Make installed the simply run 'make <command>'. If not, you can still just copy the commands from the Makefile to your terminal.

`make conda-update`  
Sets up a conda environment called `swe-motion-env` with the contents specified in [environment.yml](environment.yml). Afterwards, make sure to activate the environment.
```bash
conda activate swe-motion-env
```
`make pip-tools`  
Installs the package dependencies in your environment. The dependency specifications can be found in the [requirements/](requirements/) directory.


## Training

When a training is started 
1. Training data is downloaded from the Swedish parliament ([read more here](https://data.riksdagen.se/in-english/)), if not already present on your system. It will be kept in the [data/](data/) directory.
2. A version of [Google's MT5](https://huggingface.co/docs/transformers/model_doc/mt5) language models will be downloaded fine tuned on the training data using the PyTorch Lightning framework. In the prepped training data each example is a motion text, and it's target the motion's title.
3. There will be a saved model checkpoint for the epoch with the best validation score in the `training/logs/lightning_logs/` directory.

To list the available training parameters, run:
```bash
make train-help
```
You can use the following command to run a slimmed training loop to make sure everything behaves as intended,  but note that a dev run will not produce a model checkpoint.
```bash
make train-dev-run
```


## Output data

The raw data record above will be represented as a **row** with the following information in the output data:

| Column Name      |        Row Value |
| :--------------- | ---------------: |
| game_id          |      30582557607 |
| start_date_local |       2021-11-13 |
| start_time_local |         23:46:20 |
| end_date_local   |       2021-11-13 |
| end_time_local   |         23:50:52 |
| event            |       Live Chess |
| site             |        Chess.com |
| time_class       |            blitz |
| time_control     |              180 |
| result           |              1-0 |
| termination      |      resignation |
| eco              |              C00 |
| name             |          gripklo |
| color            |            Black |
| is_white         |                0 |
| is_black         |                1 |
| rating           |             1531 |
| is_win           |                0 |
| is_loss          |                1 |
| is_draw          |                0 |
| result_str       |             Loss |
| won_points       |              0.0 |
| opp_name         | Learning_Process |
| opp_rating       |             1599 |
