#!/bin/bash
itraj=$1
mode=$2
cd ~/continuous
. ~/qenv_bilkis/bin/activate
python3 run.py --seed $itraj --mode $mode
deactivate
