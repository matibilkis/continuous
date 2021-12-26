#!/bin/bash
itraj=$1
periods=$2
trainid=$3
cd ~/continuous
. ~/qenv_bilkis/bin/activate
python3 main_train.py --itraj $itraj --periods $periods --trainid $trainid
deactivate
