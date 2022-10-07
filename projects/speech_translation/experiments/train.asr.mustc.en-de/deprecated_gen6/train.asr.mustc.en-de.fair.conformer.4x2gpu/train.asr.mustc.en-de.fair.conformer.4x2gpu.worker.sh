#!/bin/bash

echo 'Activating venv....'
source /work/smt4/gao/tran/speech_translation/venv-fairseq-py39/bin/activate

PROJECT_NAME=speech_translation
MODEL_NAME=fair.conformer.4x2gpu
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
--lr 1e-3 \
--warmup-updates 15000 \
--lr-scheduler inverse_sqrt \
--update-freq 8 \
--max-tokens 5000 \
--max-epoch 30 \
--report-accuracy \
--epoch-split 10 \
--log-format simple \
--log-interval 100 \
--ddp-backend slowmo \
--slowmo-momentum 0.0 \
--slowmo-base-algorithm sgp
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

if [ -f CURRENT_MASTER ]; then
    CURRENT_MASTER=$(cat CURRENT_MASTER)
    RESOLVED_MASTER=$(nslookup $CURRENT_MASTER | tail -n2 | head -n1 | cut -d' ' -f2)
else
  echo 'No current master found, exiting...'
  exit
fi

echo '-------------------------'
echo "This is a worker script, running on $(hostname)."
echo "Will communicate with master $CURRENT_MASTER"
echo "Master resolves to $RESOLVED_MASTER"
echo '-------------------------'
echo "Current fairseq commit: "
echo "$(git -C $FAIRSEQ_PATH rev-parse HEAD)"
echo '-------------------------'

# https://www-i6.informatik.rwth-aachen.de/publications/download/1139/Bahar-IWSLT-2020.pdf
# Describes training of transformer in section 2.4
echo 'Running fairseq-train with following parameters...'
echo $DATA_PATH --task custom_speech_to_text --config-yaml $DATA_PATH/config_asr.espnet.yaml $MODEL_ARGS $TRAIN_ARGS $OTHER_ARGS
echo '-------------------------'



NCCL_SOCKET_IFNAME=enp5s0 WANDB_RUN_GROUP=$GROUP_NAME WANDB_NAME=$MODEL_NAME python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=4 \
  --node_rank=$1 \
  --master_port=60000 \
  --master_addr=$RESOLVED_MASTER \
  $USER_DIR/cli/train.py \
  $DATA_PATH --task custom_speech_to_text --config-yaml $DATA_PATH/config_asr.espnet.yaml $MODEL_ARGS $TRAIN_ARGS $OTHER_ARGS
