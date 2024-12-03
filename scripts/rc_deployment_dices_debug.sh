#!/bin/bash -l 

#SBATCH --job-name=dices_calssification
#SBATCH --account pop-ml 
#SBATCH --partition=debug

#SBATCH --output=logs/sbatch_log/%x_%j.out		
#SBATCH --error=logs/sbatch_err/%x_%j.err	

#SBATCH --cpus-per-task=4 
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=slack:@rk7527
#SBATCH --mail-type=ALL
#SBATCH --mem=100g

#SBATCH --time=12:00:00

conda activate finetuning
python main.py

