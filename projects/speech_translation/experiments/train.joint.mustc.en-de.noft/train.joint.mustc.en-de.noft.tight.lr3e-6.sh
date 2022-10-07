#!/bin/bash

echo 'Activating venv....'
source /work/smt4/gao/tran/speech_translation/venv-fairseq-py39/bin/activate

PROJECT_NAME=speech_translation
MODEL_NAME=joint.mustc.en-de.noft.tight.lr3e-6
GROUP_NAME=joint.mustc.en-de.noft
TAGS=joint,piv1,nomulti,tight

FAIRSEQ_PATH=/work/smt4/gao/tran/speech_translation/fairseq
MODEL_PATH=/work/smt4/gao/tran/$PROJECT_NAME/trained_models/$GROUP_NAME/$MODEL_NAME
USER_DIR=$FAIRSEQ_PATH/projects/$PROJECT_NAME

NUM_GPUS=${1:-1}

read -r -d '' MODEL_ARGS <<EOF
--data-asr /work/smt2/vtran/datasets/asr/parnia.mustc.asr.v2
--config-yaml-asr /work/smt2/vtran/datasets/asr/parnia.mustc.asr.v2/config_asr.i6approx.yaml \
--data-st /work/smt2/vtran/datasets/asr/parnia.mustc.st \
--config-yaml-st /work/smt2/vtran/datasets/asr/parnia.mustc.st/config_st.mt-bpe.yaml \
--data-mt /work/smt4/gao/tran/speech_translation/datasets/text/manual/parnia.mustc-mt.en-de.asr-bpe.src-normalized/data-bin.v2 \
--arch cascaded_st \
--asr-arch fair_s2t_transformer \
--mt-arch relpos_transformer \
--asr-model-conf $USER_DIR/experiments/train.joint.mustc.en-de.noft/asr.base.noft.yaml \
--mt-model-conf $USER_DIR/experiments/train.joint.mustc.en-de.noft/mt.base.yaml \
--criterion joint_cascaded_st \
--mt-label-smoothing 0.1 \
--asr-label-smoothing 0.1 \
--st-label-smoothing 0.1 \
--st-weight 1 \
--source-lang en \
--target-lang de \
--skip-invalid-size-inputs-valid-test \
--pivot-beam-generate 12 \
--pivot-beam 1 \
--max-source-positions 4000 \
--max-pivot-positions 75 \
--max-target-positions 75 \
--mt-checkpoint /work/smt4/gao/tran/speech_translation/trained_models/mt.mustc.en-de/src-asr-like.asr-bpe.base/checkpoints/checkpoint_best.pt \
--tight-integrated \
--tight-integrated-training-temp 1.0 \
--tight-integrated-decoding-temp 1.5
EOF

read -r -d '' TRAIN_ARGS <<EOF
--train-subset train_st \
--valid-subset dev_st \
--optimizer adam --adam-betas (0.9,0.999) --adam-eps 1e-8 --clip-norm 10.0 \
--lr 0.000003 \
--warmup-subepochs 10 \
--lr-scheduler subepoch_reduce_lr_on_plateau --lr-shrink 0.8 --lr-threshold 5e-3 --lr-patience 3 \
--weight-decay 0.0001 \
--update-freq 8 \
--batch-size 50 \
--max-tokens 4000 \
--max-epoch 6 \
--report-accuracy \
--epoch-split 10 \
--cache-manager \
--ddp-backend legacy_ddp
EOF

read -r -d '' OTHER_ARGS <<EOF
--num-workers 2 \
--no-progress-bar \
--log-file $MODEL_PATH/train.log \
--wandb-project $PROJECT_NAME \
--user-dir $USER_DIR \
--save-dir $MODEL_PATH/checkpoints \
--keep-best-checkpoints 2 \
--keep-last-subepochs 3 \
--skip-invalid-size-inputs-valid-test
EOF

mkdir -p $MODEL_PATH
cd $MODEL_PATH || exit

echo '-------------------------'
echo "Current fairseq commit: "
git -C $FAIRSEQ_PATH rev-parse HEAD
echo '-------------------------'

echo 'Running fairseq-train with following parameters...'
echo --task cascaded_speech_translation "$MODEL_ARGS" "$TRAIN_ARGS" "$OTHER_ARGS"
echo '-------------------------'

if [ "$NUM_GPUS" -eq 1 ]; then
  WANDB_TAGS=$TAGS WANDB_RUN_GROUP=$GROUP_NAME WANDB_NAME=$MODEL_NAME python3 $USER_DIR/cli/train.py \
    --task cascaded_speech_translation $MODEL_ARGS $TRAIN_ARGS $OTHER_ARGS
else
  WANDB_TAGS=$TAGS WANDB_RUN_GROUP=$GROUP_NAME WANDB_NAME=$MODEL_NAME python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 29501 \
    $USER_DIR/cli/train.py \
    --task cascaded_speech_translation $MODEL_ARGS $TRAIN_ARGS $OTHER_ARGS
fi