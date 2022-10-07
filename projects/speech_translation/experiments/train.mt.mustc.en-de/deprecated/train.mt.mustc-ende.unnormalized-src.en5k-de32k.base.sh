#!/bin/bash

echo 'Activating venv....'
source /work/smt2/vtran/virtualenvs/fairseq-py39/bin/activate

PROJECT_NAME=speech_translation
MODEL_NAME=unnormalized-src.en5k-de32k.base
GROUP_NAME=mt.mustc-ende

DATA_PATH=/work/smt2/vtran/datasets/mt/out/parnia.mustc-mt.en-de.en-5k.de-32k.src-unnormalized/data-bin
MODEL_PATH=/work/smt2/vtran/projects/$PROJECT_NAME/$GROUP_NAME/$MODEL_NAME
USER_DIR=/work/smt2/vtran/fairseq-st/projects/speech_translation

read -r -d '' MODEL_ARGS <<EOF
--arch relpos_transformer \
--share-decoder-input-output-embed \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--dropout 0.1
EOF

read -r -d '' TRAIN_ARGS <<EOF
--optimizer adam --adam-betas (0.9,0.98) --adam-eps 1e-8 --clip-norm 0.0 \
--lr 0.0003 \
--warmup-updates 4000 \
--lr-scheduler reduce_lr_on_plateau --lr-shrink 0.8 --lr-threshold 5e-3 --lr-patience 3 \
--weight-decay 0.0001 \
--update-freq 2 \
--max-tokens 8192 \
--max-epoch 100 \
--amp
EOF

read -r -d '' OTHER_ARGS <<EOF
--num-workers 3 \
--no-progress-bar \
--log-interval 50 \
--log-format simple \
--log-file $MODEL_PATH/train.log \
--wandb-project $PROJECT_NAME \
--user-dir $USER_DIR \
--save-dir $MODEL_PATH/checkpoints \
--keep-best-checkpoints 1 \
--keep-last-epochs 10
EOF

mkdir -p $MODEL_PATH
cd $MODEL_PATH

echo '-------------------------'
echo "Current fairseq commit: "
echo "$(git -C /u/vtran/work/fairseq-st rev-parse HEAD)"
echo '-------------------------'

# https://www-i6.informatik.rwth-aachen.de/publications/download/1139/Bahar-IWSLT-2020.pdf
# Describes training of transformer in section 2.4
echo 'Running fairseq-train with following parameters...'
echo $DATA_PATH --task custom_translation $MODEL_ARGS $TRAIN_ARGS $OTHER_ARGS
echo '-------------------------'

WANDB_RUN_GROUP=$GROUP_NAME WANDB_NAME=$MODEL_NAME python /work/smt2/vtran/fairseq-st/fairseq_cli/train.py \
  $DATA_PATH --task custom_translation $MODEL_ARGS $TRAIN_ARGS $OTHER_ARGS
