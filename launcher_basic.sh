#!/bin/bash
#SBATCH --account=rrg-dprecup
#SBATCH --ntasks=1
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kushal.arora@mail.mcgill.ca
#SBATCH --time=3:00:00
###########################

set -eux
$@
