#!/bin/bash

echo 'Activating venv....'
source /work/smt4/gao/tran/speech_translation/venv-fairseq-py39/bin/activate

PROJECT_NAME=speech_translation
MODEL_NAME=fair.conformer.4gpu
GROUP_NAME=asr.mustc.en-de

FAIRSEQ_PATH=/work/smt4/gao/tran/speech_translation/fairseq
DATA_PATH=/work/smt2/vtran/datasets/asr/parnia.mustc.asr.v2
MODEL_PATH=/work/smt4/gao/tran/$PROJECT_NAME/trained_models/$GROUP_NAME/$MODEL_NAME
USER_DIR=$FAIRSEQ_PATH/projects/$PROJECT_NAME

read -r -d '' MODEL_ARGS <<EOF
--arch fair_conformer \
--share-decoder-input-output-embed \
--criterion ctc_label_smoothed_cross_entropy --label-smoothing 0.1 \
--ctc-loss --ctc-weight 0.3
EOF

read -r -d '' TRAIN_ARGS <<EOF
--train-subset train_asr \
--valid-subset dev_asr \
--optimizer adam --clip-norm 10.0 \
--lr 8e-4 \
--warmup-updates 25000 \
--lr-scheduler inverse_sqrt \
--update-freq 8 \
--max-tokens 5000 \
--max-epoch 30 \
--report-accuracy \
--epoch-split 10 \
--log-format simple \
--log-interval 100 \
--ddp-backend fully_sharded
EOF

read -r -d '' OTHER_ARGS <<EOF
--data-buffer-size 80 \
--num-workers 2 \
--no-progress-bar \
--log-file $MODEL_PATH/train.log \
--wandb-project $PROJECT_NAME \
--user-dir $USER_DIR \
--save-dir $MODEL_PATH/checkpoints \
--keep-best-checkpoints 2 \
--keep-last-subepochs 10 \
--max-source-positions 4000 \
--max-target-positions 75 \
--skip-invalid-size-inputs-valid-test
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
echo $DATA_PATH --task custom_speech_to_text --config-yaml $DATA_PATH/config_asr.espnet.yaml $MODEL_ARGS $TRAIN_ARGS $OTHER_ARGS
echo '-------------------------'



NCCL_P2P_DISABLE=1 WANDB_RUN_GROUP=$GROUP_NAME WANDB_NAME=$MODEL_NAME python -m torch.distributed.launch --nproc_per_node=4 \
  $USER_DIR/cli/train.py \
  $DATA_PATH --task custom_speech_to_text --config-yaml $DATA_PATH/config_asr.espnet.yaml $MODEL_ARGS $TRAIN_ARGS $OTHER_ARGS
