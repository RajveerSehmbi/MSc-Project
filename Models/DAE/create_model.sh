#!/bin/bash
#SBATCH --partition=gpgpuC
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rs218
#SBATCH --output=create_model%j.out

export PATH=/vol/bitbucket/${USER}/fullenv/bin/:$PATH

source activate
source /vol/cuda/12.0.0/setup.sh
uptime

python3 /vol/bitbucket/rs218/Project/Models/DAE/create_model.py

/usr/bin/nvidia-smi