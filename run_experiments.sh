#!/bin/bash

# SLURM OPTIONS
#SBATCH --partition=gpu-h100
#SBATCH --time=72:00:00         # Time limit for the job
#SBATCH --job-name=iba   # Name of your job
#SBATCH --error=iba.err
#SBATCH --output=iba.out
#SBATCH --nodes=1               # Number of nodes you want to run your process on
#SBATCH --ntasks-per-node=24     # Number of CPU cores
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh 

environment_name="fl"

if ! conda info --envs | grep -q "^${environment_name}"; then
  conda create -n ${environment_name} python=3.12 -y
fi
conda activate $environment_name

pip install -r requirements.txt

PYTHONPATH=. python experiments/run_parallel.py --config experiments/configs/iba_base.yml 

PYTHONPATH=. python experiments/run_parallel.py --config experiments/configs/iba_clip.yml 

wait

