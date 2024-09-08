#!/bin/bash
#SBATCH --partition=gpgpuC
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rs218
#SBATCH --output=DAE%j.out

export PATH=/vol/bitbucket/rs218/fullenv/bin/:$PATH

source activate
source /vol/cuda/12.0.0/setup.sh
uptime

python /vol/bitbucket/rs218/MSc-Project/Models/Classifier/train_final.py train