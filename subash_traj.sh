#!/bin/bash
itraj=$1
periods=$2
ppp=$3
seed=$4
cd ~/continuous
. ~/qenv_bilkis/bin/activate
python3 main_traj.py --periods $periods --itraj $itraj --ppp $ppp --seed $seed
deactivate
