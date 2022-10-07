#!/bin/bash

echo 'Activating venv....'
source /work/smt2/vtran/virtualenvs/fairseq-py39/bin/activate

PROJECT_NAME=speech_translation
MODEL_NAME=asr.mustc.en-de.src6000
GROUP_NAME=asr.mustc

DATA_PATH=/work/smt2/vtran/datasets/asr/parnia.mustc.asr.nofilter
MODEL_PATH=/work/smt2/vtran/projects/$PROJECT_NAME/$GROUP_NAME/$MODEL_NAME
USER_DIR=/work/smt2/vtran/fairseq-st/projects/speech_translation

read -r -d '' MODEL_ARGS <<EOF
--arch relpos_s2t_transformer \
--share-decoder-input-output-embed \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--dropout 0.1 \
--encoder-layers 12 \
--decoder-layers 6
EOF

read -r -d '' TRAIN_ARGS <<EOF
--train-subset train_asr \
--valid-subset dev_asr \
--optimizer adam --adam-betas (0.9,0.98) --adam-eps 1e-8 --clip-norm 10.0 \
--lr 0.0003 \
--warmup-subepochs 10 \
--lr-scheduler subepoch_reduce_lr_on_plateau --lr-shrink 0.8 --lr-threshold 5e-3 --lr-patience 3 \
--weight-decay 0.0001 \
--update-freq 8 \
--batch-size 200 \
--max-tokens 10000 \
--max-epoch 100 \
--report-accuracy \
--epoch-split 20 \
--log-format simple \
--log-interval 100 \
--amp
EOF

read -r -d '' OTHER_ARGS <<EOF
--num-workers 2 \
--no-progress-bar \
--log-file $MODEL_PATH/train.log \
--wandb-project $PROJECT_NAME \
--user-dir $USER_DIR \
--save-dir $MODEL_PATH/checkpoints \
--keep-best-checkpoints 10 \
--keep-last-epochs 1 \
--max-source-positions 6000 \
--max-target-positions 100 \
--skip-invalid-size-inputs-valid-test
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
echo $DATA_PATH --task cached_speech_to_text --config-yaml /work/smt2/vtran/datasets/asr/parnia.mustc.asr.nofilter/config_asr.yaml $MODEL_ARGS $TRAIN_ARGS $OTHER_ARGS
echo '-------------------------'

WANDB_RUN_GROUP=$GROUP_NAME WANDB_NAME=$MODEL_NAME python /work/smt2/vtran/fairseq-st/projects/speech_translation/cli/train.py \
  $DATA_PATH --task cached_speech_to_text --config-yaml /work/smt2/vtran/datasets/asr/parnia.mustc.asr.nofilter/config_asr.yaml $MODEL_ARGS $TRAIN_ARGS $OTHER_ARGS
