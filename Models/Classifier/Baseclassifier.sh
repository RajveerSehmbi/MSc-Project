#!/bin/bash
#SBATCH --partition=gpgpuC
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rs218
#SBATCH --output=Baseclassifier%j.out

export PATH=/vol/bitbucket/rs218/fullenv/bin/:$PATH

source activate

python train_nnGPUBase.py