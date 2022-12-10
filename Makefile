# Arcane incantation to print all the other targets, from https://stackoverflow.com/a/26339924
help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

conda-clean-update:
	conda env remove --name swe-parl-mot-summarization
	conda clean --yes --all --force-pkgs-dirs
	make conda-update

conda-update:
	conda env update --prune -f environment.yml
	echo "Now run:\nconda activate swe-parl-mot-summarization"

pip-tools:
	pip install pip-tools
	pip-compile requirements/prod.in && pip-compile requirements/dev.in
	pip-sync requirements/prod.txt requirements/dev.txt

training-dataset:
	python -m training_dataset_downloader

# Example training command
train-mt5:
	PYTHONPATH=. python training/run_experiment.py --max_epochs=3 --gpus=1 --batch_size=8 --num_workers=4 --data_class=SweParliamentMotionsDataModule --model_class=t5

# Overfit on single batch
overfit:
	PYTHONPATH=. python training/run_experiment.py --max_epochs=10 --gpus=0 --accelerator=ddp --num_workers=20 --data_class=SweParliamentMotionsDataModule --model_class=t5  --overfit_batches=1

# Lint
lint:
	tasks/lint.sh
