#!/bin/bash
#SBATCH -c 10
#SBATCH --constraint="[type_b]"
#SBATCH --time=720:00:00
#SBATCH --output=/home/nsborodin/Maksim/output/RuTF-IDF_SVD.log

source /home/nsborodin/Maksim/my_env/bin/activate
python RuTF-IDF_SVD.py
