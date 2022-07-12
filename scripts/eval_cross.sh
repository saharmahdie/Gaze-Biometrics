#!/bin/env bash

METHOD=${1-"score-level"}
DATASET=${2-"bio-tex"}
CLF=$3


ITERS=100
for i in $(seq 1 $ITERS); do
  python3 evaluation.py --method $METHOD --dataset $DATASET --clf $CLF --ul 50 --seed $i
done