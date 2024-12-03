#!/bin/bash -l
conda activate finetuning
SCRIPT_DIR=$(dirname "$(realpath "$0")")
python "$SCRIPT_DIR/setup.py" --project_dir "$SCRIPT_DIR"