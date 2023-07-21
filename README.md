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


## Raw data

There's new data added to the API calls over time. At the time of this documentation a typical game would look like this:

```yaml
{
  "url": "https://www.chess.com/game/live/30582557607",
  "pgn": "[Event \"Live Chess\"]\n[Site \"Chess.com\"]\n[Date \"2021.11.13\"]\n[Round \"-\"]\n[White \"Learning_Process\"]\n[Black \"gripklo\"]\n[Result  \"1-0\"]\n[CurrentPosition \"4rk2/p4ppp/8/8/1r1P4/3KBN2/BPn2PPP/1R5R b - -\"]\n[Timezone \"UTC\"]\n[ECO \"C00\"]\n[ECOUrl \"https://www.chess.com/openings/French-Defense-Normal-Variation\"]\n[UTCDate \"2021.11.13\"]\n[UTCTime \"22:46:20\"]\n[WhiteElo \"1599\"]\n[BlackElo \"1531\"]\n[TimeControl \"180\"]\n[Termination \"Learning_Process won by resignation\"]\n[StartTime \"22:46:20\"]\n[EndDate \"2021.11.13\"]\n[EndTime \"22:50:52\"]\n[Link \"https://www.chess.com/game/live/30582557607\"]\n\n1. d4 {[%clk 0:02:57]} 1... e6 {[%clk 0:02:58.7]} 2. e4 {[%clk 0:02:56]} 2... d5 {[%clk 0:02:56.8]} 3. exd5 {[%clk 0:02:54]} 3... exd5 {[%clk 0:02:55.5]} 4. c4 {[%clk 0:02:53.9]} 4... Nf6 {[%clk 0:02:53.8]} 5. cxd5 {[%clk 0:02:52.7]} 5... Nxd5 {[%clk 0:02:52.4]} 6. Nc3 {[%clk 0:02:51.6]} 6... Be6 {[%clk 0:02:47.5]} 7. Nf3 {[%clk 0:02:50]} 7... Nc6 {[%clk 0:02:42]} 8. Qb3 {[%clk 0:02:45.8]} 8... Be7 {[%clk 0:02:25.5]} 9. Qxb7 {[%clk 0:02:44]} 9... Ncb4 {[%clk 0:01:59.5]} 10. Bb5+ {[%clk 0:02:39.5]} 10... c6 {[%clk 0:01:45.4]} 11. Bxc6+ {[%clk 0:02:37.6]} 11... Kf8 {[%clk 0:01:22.7]} 12. Nxd5 {[%clk 0:02:22.6]} 12... Nc2+ {[%clk 0:01:11.2]} 13. Ke2 {[%clk 0:02:20.8]} 13... Rb8 {[%clk 0:01:00.1]} 14. Qxe7+ {[%clk 0:02:15.1]} 14... Qxe7 {[%clk 0:00:56.5]} 15. Nxe7 {[%clk 0:02:14]} 15... Kxe7 {[%clk 0:00:54.7]} 16. Rb1 {[%clk 0:02:06]} 16... Bxa2 {[%clk 0:00:52.6]} 17. Bf4 {[%clk 0:02:00.7]} 17... Rbc8 {[%clk 0:00:46.4]} 18. Bb7 {[%clk 0:01:49.1]} 18... Rc4 {[%clk 0:00:35]} 19. Bd5 {[%clk 0:01:45.5]} 19... Rb4 {[%clk 0:00:26.8]} 20. Bxa2 {[%clk 0:01:43.3]} 20... Re8 {[%clk 0:00:20.8]} 21. Be3 {[%clk 0:01:36.5]} 21... Kf8 {[%clk 0:00:17.9]} 22. Kd3 {[%clk 0:01:33.8]} 1-0\n",
  "time_control": "180",
  "end_time": 1636843852,
  "rated": true,
  "tcn": "lB0SmCZJCJSJkA!TAJTJbs6Sgv5Qdr90rXQzfHYQHQ89sJzkem45X070J090abSicD56QX6AXJAzJi?8Du09mt",
  "uuid": "84fcf053-44d3-11ec-aecf-09ff3c010001",
  "initial_setup": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "fen": "4rk2/p4ppp/8/8/1r1P4/3KBN2/BPn2PPP/1R5R b - -",
  "time_class": "blitz",
  "rules": "chess",
  "white":
    {
      "rating": 1599,
      "result": "win",
      "@id": "https://api.chess.com/pub/player/learning_process",
      "username": "Learning_Process",
      "uuid": "eb465df6-dfb8-11e4-802e-000000000000",
    },
  "black":
    {
      "rating": 1531,
      "result": "resigned",
      "@id": "https://api.chess.com/pub/player/gripklo",
      "username": "gripklo",
      "uuid": "d6e0aeae-363b-11eb-9a7a-678f2547de7f",
    },
}
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
