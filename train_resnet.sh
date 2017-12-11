#!/bin/bash
python train_resnet18.py --fsize 32 --batch-size 8
sleep 5s
python train_resnet18.py --fsize 48 --batch-size 8
sleep 5s
python train_resnet18.py --fsize 64 --batch-size 8

