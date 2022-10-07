#!/bin/bash
MUSTC_ROOT=/work/smt2/vtran/datasets/mustc

cd $(dirname "$0")

python prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task asr \
  --vocab-type unigram --vocab-size 5000 --threads 4 --chunks-per-thread 100
