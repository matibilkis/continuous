#!/bin/bash
itraj=$1
seed=$2
periods=$3
ppp=$4
cd ~/continuous
. ~/qenv_bilkis/bin/activate
python3 main_traj.py --ppp $ppp --periods $periods --itraj $itraj --seed $seed
echo "integration done!"
deactivate
