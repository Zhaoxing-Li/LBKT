#!/bin/bash

# SLURM config
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH --mem=8g

#SBATCH -p "ug-gpu-small"
#SBATCH --qos="long-low-prio"
#SBATCH -t 05-00:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user joe.shepherd@durham.ac.uk

# venv config
virtualenv env
env/bin/pip install -r requirements.txt

# execute
env/bin/python train.py
