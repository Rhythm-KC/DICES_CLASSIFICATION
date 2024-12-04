#!/bin/bash -l
SCRIPT_DIR=$(dirname "$(realpath "$0")")

local=false
debug=false
deploy=false

while getopts "lds" opt; do
    case "$opt" in
    l)  # Flag -l (local)
        local=true
        ;;
    d)  # Flag -d (debug)
        debug=true
        ;;
    s) # Flag -s (deploy)
        deploy=true
        ;;
    *)  # Invalid flag
        echo "Usage: $0 [-l] [-d] [-s]. 
            -l: To run the script locally.
            -d: To deploy to slurm in an debug enviromnet
            -s: To deploy to slurm in a normal environemnt"
        exit 1
        ;;
    esac
done

conda activate finetuning
cd "$SCRIPT_DIR"

if $local; then
    python main.py
fi

if $debug; then
    sbatch srcipts/rc_deployment_dices_debug.sh
fi

if $deploy; then
    sbatch srcipts/rc_deployment_dices.sh
fi