#!/bin/bash
#SBATCH -c 10
#SBATCH --time=720:00:00
#SBATCH --output=/home/nsborodin/Maksim/output/SvSVD.log

source /home/nsborodin/Maksim/my_env/bin/activate
python3 SvSVD.py
