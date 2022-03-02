#!/bin/bash
itraj=$1
cd ~/continuous
. ~/qenv_bilkis/bin/activate
python3 run.py --seed $itraj
deactivate
