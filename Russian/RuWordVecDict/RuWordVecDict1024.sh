#!/bin/bash
#SBATCH --constraint="[type_b]"
#SBATCH -c 10
#SBATCH --time=720:00:00
#SBATCH --output=/home/nsborodin/Maksim/output/RuWordVecDict.log

source /home/nsborodin/Maksim/my_env/bin/activate
python RuWordVecDict.py 1024 "/home/nsborodin/Maksim/Russian/RuTF-IDF_SVD/WORD_LIST.npy" "/home/nsborodin/Maksim/Russian/RuPreprocessed/U.npy" "/home/nsborodin/Maksim/Russian/RuPreprocessed/Sigma.npy"
