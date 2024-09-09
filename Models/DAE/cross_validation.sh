#!/bin/bash
#SBATCH --partition=gpgpuC
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rs218
#SBATCH --output=cv%j.out

export PATH=/vol/bitbucket/${USER}/fullenv/bin/:$PATH

source activate
source /vol/cuda/12.0.0/setup.sh
uptime

python3 /vol/bitbucket/rs218/MSc-Project/Models/DAE/cross_validation.py

/usr/bin/nvidia-smi
