help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

conda-clean-update:
	conda env remove --name swe-motion-env
	conda clean --yes --all --force-pkgs-dirs
	make conda-update

conda-update:
	conda env update --prune -f environment.yml
	echo "Now run:\nconda activate swe-motion-env"

pip-tools:
	pip install pip-tools
	pip-compile requirements/prod.in && pip-compile requirements/dev.in
	pip-sync requirements/prod.txt requirements/dev.txt

data-pipeline:
	PYTHONPATH=. python training_data_pipeline/pipeline.py

# Overfit on single batch
overfit:
	PYTHONPATH=. python training/run_experiment.py --max_epochs=300 --num_workers=0 --data_class=MotionsDataModule --model_class=MT5 --overfit_batches=1 --lr=0.001 --early_stopping=50

minimal-training:
	PYTHONPATH=. python training/run_experiment.py --max_epochs=2 --num_workers=0 --data_class=MotionsDataModule --model_class=MT5 --fast_dev_run=True  --data_frac=0.001

# Lint
lint:
	tasks/lint.sh

# Test
test:
	PYTHONPATH=. tasks/test.sh

unit-test:
	PYTHONPATH=. pytest -s .
