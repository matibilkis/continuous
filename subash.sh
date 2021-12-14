#!/bin/bash
points=$1
time=$2
indextraj=$(ls -1q | wc -l)
echo $indextraj
cd ~/dynamo
. ~/qenv_bilkis/bin/activate
python3 main.py --N points --T time --itraj indextraj
deactivate
