#!/bin/bash
ppp=$1
periods=$2
cd ~/continuous
. ~/qenv_bilkis/bin/activate
indextraj=$(ls -1q | wc -l)
echo $ppp
echo ppp
echo $periods
echo indextraj
echo $indextraj
python3 main.py --ppp ppp --periods periods --itraj indextraj
deactivate