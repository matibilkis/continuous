#!/bin/bash
mode=$1
cd ~/continuous
. ~/qenv_bilkis/bin/activate
python3 analysis/analyze.py --mode $mode
deactivate
