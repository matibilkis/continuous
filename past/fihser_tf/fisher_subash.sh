#!/bin/bash
itraj=$1
cd ~/continuous
. ~/qenv_bilkis/bin/activate
python3 fisher_main.py --seed $itraj
deactivate
