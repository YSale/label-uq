#!/bin/sh
#source venv/bin/activate
#conda activate env

name='fmnist'
echo "run $name"
python holdout.py --data $name --epochs 5 --epochs_hold 5 --hold_rate 0.995 --runs 5 --num_members 5 --seed 2 

name='cifar'
echo "run $name"
python holdout.py --data $name --epochs 20 --epochs_hold 20 --hold_rate 0.9 --runs 5 --num_members 5 --seed 2 