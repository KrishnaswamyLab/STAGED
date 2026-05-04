#!/bin/bash

#SBATCH --job-name=axalotl
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --output=./logs/slurm/axalotl/afdb/%x_%j.out
#SBATCH --error=./logs/slurm/axalotl/afdb/%x_%j.err
#SBATCH --mail-type=REQUEUE,FAIL,TIME_LIMIT


cd /home/jcr222/workspace/STAGED
ml uv
source .venv/bin/activate

date
hostname
pwd

uv run python src/main.py --mode train --config /home/jcr222/workspace/STAGED/src/config/ode_config.yaml