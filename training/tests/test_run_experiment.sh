#!/bin/bash
FAILURE=false

PYTHONPATH=. python training/run_experiment.py --max_epochs=3 --batch_size=1 --num_workers=4 --data_class=MotionsDataModule --model_class=MT5 --overfit_batches=1 || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Test for run_experiment.py failed"
  exit 1
fi
echo "Test for run_experiment.py passed"
exit 0
