#!/bin/bash
ppp=$1
periods=$2
cd ~/continuous
. ~/qenv_bilkis/bin/activate
indextraj=$(ls -1q | wc -l)
python3 main_traj.py --ppp ppp --periods periods --itraj indextraj
python3 main_train.py --itraj indextraj
deactivate
