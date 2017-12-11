#!/bin/bash
python train_mobilenet.py --alpha 1.0 --batch-size 8
sleep 5s
python train_mobilenet.py --alpha 0.75 --batch-size 8
sleep 5s
python train_mobilenet.py --alpha 0.5 --batch-size 8
sleep 5s
python train_mobilenet.py --alpha 0.25 --batch-size 8
