#!/bin/bash
#SBATCH --partition=gpgpuC
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rs218
#SBATCH --output=dae%j.out

export PATH=/vol/bitbucket/${USER}/fullenv/bin/:$PATH

source activate
source /vol/cuda/12.0.0/setup.sh
uptime

python3 /vol/bitbucket/rs218/MSc-Project/Data/SQLite/DAE_transform.py

/usr/bin/nvidia-smi
