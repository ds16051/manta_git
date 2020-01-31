#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --time 0-00:30
#SBATCH --mail-type=END
#SBATCH --mem 64GB
#SBATCH --gres gpu:1


module purge

module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

python train.py
