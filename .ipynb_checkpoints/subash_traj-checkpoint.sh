#!/bin/bash
itraj=$1
periods=$2
cd ~/continuous
. ~/qenv_bilkis/bin/activate
python3 main_traj.py --periods $periods --itraj $itraj
python3 main_train.py --itraj $itraj
deactivate
