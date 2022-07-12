#!/bin/env bash

CLF=$1
TRAIN_SAMPLES=$2
TEST_SAMPLES=$3
METHOD=${4-"score-level"}
RBNF_K=${5-32}


ITERS=5
for i in $(seq 1 $ITERS); do
  python3 evaluation.py --method $METHOD --dataset whl --ivt_threshold 150 --clf $CLF  --whl_train_samples $TRAIN_SAMPLES --whl_test_samples $TEST_SAMPLES --rbfn_k $RBNF_K --seed $i
done