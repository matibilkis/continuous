#!/bin/bash
itraj=$1
trainid=$2
periods=$3
ppp=$4

cd ~/continuous
. ~/qenv_bilkis/bin/activate
python3 main_train.py --ppp $ppp --periods $periods --itraj $itraj --trainid $trainid
echo "TRAINING done!"
deactivate
