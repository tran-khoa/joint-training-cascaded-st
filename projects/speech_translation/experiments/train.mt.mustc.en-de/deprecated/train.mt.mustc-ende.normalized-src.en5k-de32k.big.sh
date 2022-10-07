#!/bin/bash

echo 'Activating venv....'
source /work/smt4/gao/tran/speech_translation/venv-fairseq-py39/bin/activate

PROJECT_NAME=speech_translation
MODEL_NAME=normalized-src.en5k-de32k.big
GROUP_NAME=mt.mustc.en-de

FAIRSEQ_PATH=/work/smt4/gao/tran/speech_translation/fairseq
DATA_PATH=/work/smt4/gao/tran/speech_translation/datasets/text/out/parnia.mustc-mt.en-de.en-5k.de-32k.src-normalized/data-bin
MODEL_PATH=/work/smt4/gao/tran/$PROJECT_NAME/trained_models/$GROUP_NAME/$MODEL_NAME
USER_DIR=$FAIRSEQ_PATH/projects/$PROJECT_NAME

read -r -d '' MODEL_ARGS <<EOF
--arch relpos_transformer \
--share-decoder-input-output-embed \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--dropout 0.3 \
--encoder-embed-dim 1024 \
--decoder-embed-dim 1024 \
--encoder-ffn-embed-dim 4096 \
--decoder-ffn-embed-dim 4096 \
--encoder-attention-heads 16 \
--decoder-attention-heads 16
EOF

read -r -d '' TRAIN_ARGS <<EOF
--optimizer adam --adam-betas (0.9,0.98) --adam-eps 1e-8 --clip-norm 0.0 \
--lr 0.0003 \
--warmup-updates 4000 \
--lr-scheduler reduce_lr_on_plateau --lr-shrink 0.9 --lr-threshold 5e-3 --lr-patience 20 \
--weight-decay 0.0001 \
--update-freq 4 \
--max-tokens 4096 \
--max-epoch 100
EOF

read -r -d '' OTHER_ARGS <<EOF
--data-buffer-size 20 \
--num-workers 2 \
--no-progress-bar \
--log-interval 50 \
--log-format simple \
--log-file $MODEL_PATH/train.log \
--wandb-project $PROJECT_NAME \
--user-dir $USER_DIR \
--save-dir $MODEL_PATH/checkpoints \
--keep-best-checkpoints 2 \
--keep-last-epochs 1 \
--save-interval 1 \
--save-interval-updates 10000 \
--keep-interval-updates 10
EOF

mkdir -p $MODEL_PATH
cd $MODEL_PATH

echo '-------------------------'
echo "Current fairseq commit: "
echo "$(git -C $FAIRSEQ_PATH rev-parse HEAD)"
echo '-------------------------'

# https://www-i6.informatik.rwth-aachen.de/publications/download/1139/Bahar-IWSLT-2020.pdf
# Describes training of transformer in section 2.4
echo 'Running fairseq-train with following parameters...'
echo $DATA_PATH --task custom_translation $MODEL_ARGS $TRAIN_ARGS $OTHER_ARGS
echo '-------------------------'

WANDB_RUN_GROUP=$GROUP_NAME WANDB_NAME=$MODEL_NAME python $USER_DIR/cli/train.py  \
  $DATA_PATH --task custom_translation $MODEL_ARGS $TRAIN_ARGS $OTHER_ARGS
