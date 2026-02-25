#!/bin/bash
#SBATCH --partition=gpu-h100
#SBATCH --time=72:00:00
#SBATCH --job-name=exp
#SBATCH --error=job-%j.out
#SBATCH --output=job-%j.out
#SBATCH --nodes=1
#SBATCH --mem=120GB
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

module load Anaconda3
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh

environment_name="backdoor"

# Create the Conda environment if it doesn't exist
if ! conda info --envs | grep -q "^${environment_name}"; then
    conda create -n ${environment_name} python=3.12 -y
fi

# Activate the environment
conda activate $environment_name


# Set the repo root as Python path
export PYTHONPATH=/Utilisateurs/mdif01/llm/BackdoorFramework:$PYTHONPATH


# Install dependencies from requirements.txt
# pip install -r requirements.txt

# Run your script
# python experiments/run_parallel.py --config experiments/configs/a3fl_clip.yml   
# python experiments/run_parallel.py --config experiments/configs/base_parallel.yml  
# python experiments/run_parallel.py --config experiments/configs/iba_base.yml  
# python experiments/run_parallel.py --config experiments/configs/mr_base.yml  
# python experiments/run_parallel.py --config experiments/configs/mr_krum.yml          
# python experiments/run_parallel.py --config experiments/configs/no_attack.yml

# python experiments/run_parallel.py --config experiments/configs/a3fl_flame_new.yml  

python experiments/run_parallel.py --config experiments/configs/a3fl_flame_layer.yml  

# python experiments/run_parallel.py --config experiments/configs/config.yml         
# python experiments/run_parallel.py --config experiments/configs/iba_clip.yml  
# python experiments/run_parallel.py --config experiments/configs/mr_clip.yml  
# python experiments/run_parallel.py --config experiments/configs/neurotoxin_base.yml

wait
