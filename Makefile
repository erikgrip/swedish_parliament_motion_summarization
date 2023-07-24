help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

# Setup
conda-clean-update:
	conda env remove --name swe-motion-env
	conda clean --yes --all --force-pkgs-dirs
	make conda-update

conda-update:
	conda env update --prune -f environment.yml
	echo "Now run:\nconda activate swe-motion-env"

pip-sync:
	pip-sync requirements/prod.txt requirements/dev.txt

pip-tools:
	pip install pip-tools
	pip-compile requirements/prod.in && pip-compile requirements/dev.in
	make pip-sync


# Training
train-help:
	PYTHONPATH=. python training/run_experiment.py --help

train-overfit:
	PYTHONPATH=. python training/run_experiment.py --max_epochs=300 --data_class=MotionsDataModule --model_class=MT5 --overfit_batches=1 --lr=0.001 --early_stopping=50

train-dev-run:
	PYTHONPATH=. python training/run_experiment.py --max_epochs=2 --data_class=MotionsDataModule --model_class=MT5 --fast_dev_run=True  --data_frac=0.001

tensorboard:
	tensorboard --logdir training/logs/lightning_logs


# Testing and linting
lint:
	tasks/lint.sh

test:
	PYTHONPATH=. tasks/test.sh

unit-test:
	PYTHONPATH=. pytest -s .


# Prediction
build-sample-app-image:
	./api_server/build_app_image.sh hf erikgrip2/mt5-finetuned-for-motion-title

run-app:
	docker stop app_container || true && docker rm app_container || true
	docker run -p 8000:8000 --name app_container motion_title_app


