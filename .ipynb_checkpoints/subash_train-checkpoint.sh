#!/bin/bash
itraj=$1
periods=$2
trainid=$3
ppp=$4
oneparam=$5
cd ~/continuous
. ~/qenv_bilkis/bin/activate
python3 main_train.py --itraj $itraj --periods $periods --trainid $trainid --ppp $ppp --oneparam $oneparam
deactivate
