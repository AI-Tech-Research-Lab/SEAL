#!/bin/bash

# Nombre del trabajo
#SBATCH --job-name="i16-multi"
#SBATCH --output=00_multi_fixed_i16.txt

# Cola de trabajo
#SBATCH --partition=gpus

# Solicitud de gpus
#SBATCH --nodelist=n7
#SBATCH --gres=gpu:quadro_rtx_8000:1

# module load cuda/12.3
eval "$(conda shell.bash hook)"
conda activate continual-nas

# Run the code
python scripts/02_multi_obj_nas/efficient_fixed.py --run_config scripts/02_multi_obj_nas/experiments/02_imagenet16_fixed.yml
