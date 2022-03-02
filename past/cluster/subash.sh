#!/bin/bash
itraj=$1
periods=$2
ppp=$3
cd ~/continuous
. ~/qenv_bilkis/bin/activate
python3 landscape.py --itraj $itraj --periods $periods --ppp $ppp
deactivate
