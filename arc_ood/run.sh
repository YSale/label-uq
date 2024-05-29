#!/bin/sh
source venv/bin/activate
#echo "Running seed 1"
#python3 train.py --data cifar --epochs 30 --seed 1
# echo "Running seed 2"
# python3 train.py --data cifar --epochs 30 --seed 2
# echo "Running seed 3"
# python3 train.py --data cifar --epochs 30 --seed 3
echo "Running seed 4"
python3 train.py --data cifar --epochs 30 --seed 4
echo "Running seed 5"
python3 train.py --data cifar --epochs 30 --seed 5