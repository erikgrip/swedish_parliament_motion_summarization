#!/bin/bash
FAILURE=false

PYTHONPATH=. python training/run_experiment.py --max_epochs=1 --accelerator=cpu --data_fraction=0.0001 || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Test for run_experiment.py failed"
  exit 1
fi
echo "Test for run_experiment.py passed"
exit 0
