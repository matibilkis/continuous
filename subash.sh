#!/bin/bash
itraj=$1
cd ~/continuous
. ~/qenv_bilkis/bin/activate
python3 meta_meta.py --seed $itraj
deactivate
