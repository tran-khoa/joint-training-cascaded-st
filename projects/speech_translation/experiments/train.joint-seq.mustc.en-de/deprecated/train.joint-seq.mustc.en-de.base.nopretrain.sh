#!/bin/bash

echo 'Activating venv....'
source /work/smt2/vtran/virtualenvs/fairseq-py39/bin/activate

PROJECT_NAME=speech_translation
MODEL_NAME=joint-seq.mustc.en-de.base.nopretrain
GROUP_NAME=joint-seq.mustc

MODEL_PATH=/work/smt2/vtran/projects/$PROJECT_NAME/$GROUP_NAME/$MODEL_NAME
USER_DIR=/work/smt2/vtran/fairseq-st/projects/speech_translation

read -r -d '' MODEL_ARGS <<EOF
--data-asr /work/smt2/vtran/datasets/asr/parnia.mustc.asr
--config-yaml-asr /work/smt2/vtran/datasets/asr/parnia.mustc.asr/config_asr.yaml \
--data-st /work/smt2/vtran/datasets/asr/parnia.mustc.st \
--config-yaml-st /work/smt2/vtran/datasets/asr/parnia.mustc.st/config_st.yaml \
--data-mt /work/smt2/vtran/datasets/mt/out/parnia.mustc-mt.en-de.en-5k.de-32k.src-normalized/data-bin \
--arch cascaded_st \
--asr-arch relpos_s2t_transformer \
--mt-arch relpos_transformer \
--asr-model-conf $USER_DIR/experiments/train.joint-seq.mustc.en-de/asr.base.yaml \
--mt-model-conf $USER_DIR/experiments/train.joint-seq.mustc.en-de/mt.base.yaml \
--criterion joint_cascaded_st \
--mt-label-smoothing 0.1 \
--asr-label-smoothing 0.1 \
--st-label-smoothing 0.1 \
--st-weight 1 \
--source-lang en \
--target-lang de \
--skip-invalid-size-inputs-valid-test
EOF

read -r -d '' TRAIN_ARGS <<EOF
--train-subset train_st,train_asr,train \
--valid-subset dev_st \
--optimizer adam --adam-betas (0.9,0.98) --adam-eps 1e-8 --clip-norm 10.0 \
--lr 0.0003 \
--warmup-subepochs 10 \
--lr-scheduler subepoch_reduce_lr_on_plateau --lr-shrink 0.8 --lr-threshold 5e-3 --lr-patience 3 \
--weight-decay 0.0001 \
--update-freq 8 \
--batch-size 200 \
--max-tokens 5000 \
--max-epoch 100 \
--report-accuracy \
--epoch-split 20
EOF

read -r -d '' OTHER_ARGS <<EOF
--num-workers 1 \
--no-progress-bar \
--log-file $MODEL_PATH/train.log \
--wandb-project $PROJECT_NAME \
--user-dir $USER_DIR \
--save-dir $MODEL_PATH/checkpoints \
--keep-best-checkpoints 10 \
--keep-last-epochs 1 \
--max-source-positions 4000 \
--max-pivot-positions 100 \
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
echo --task cascaded_speech_translation $MODEL_ARGS $TRAIN_ARGS $OTHER_ARGS
echo '-------------------------'

WANDB_RUN_GROUP=$GROUP_NAME WANDB_NAME=$MODEL_NAME python /work/smt2/vtran/fairseq-st/projects/speech_translation/cli/train.py \
  --task cascaded_speech_translation $MODEL_ARGS $TRAIN_ARGS $OTHER_ARGS
