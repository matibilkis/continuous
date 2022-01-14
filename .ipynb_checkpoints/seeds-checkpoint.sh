#!/bin/bash
seed=$1
cd ~/continuous
. ~/qenv_bilkis/bin/activate
python3 seeds.py --seed $seed
deactivate
