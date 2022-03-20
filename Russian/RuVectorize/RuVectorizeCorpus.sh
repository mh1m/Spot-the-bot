#!/bin/bash
#SBATCH --constraint="[type_b]"
#SBATCH -c 10
#SBATCH --time=720:00:00
#SBATCH --output=/home/nsborodin/Maksim/output/RuVectorizeCorpus.log

source /home/nsborodin/Maksim/my_env/bin/activate
python RuVectorizeCorpus.py
