#!/bin/bash

export PATH=/vol/bitbucket/rs218/fullenv/bin/:$PATH

source activate

python train_nnCPU.py trainPWdeepDAEtransformed
