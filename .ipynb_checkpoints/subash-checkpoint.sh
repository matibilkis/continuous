#!/bin/bash
itraj=$1
periods=$2
ppp=$3
cd ~/continuous
. ~/qenv_bilkis/bin/activate
python3 main_traj.py --ppp $ppp --periods $periods --itraj $itraj
echo "integration done!"
deactivate
