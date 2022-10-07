#!/bin/bash

echo 'Activating venv....'
source /work/smt4/gao/tran/speech_translation/venv-fairseq-py39/bin/activate

PROJECT_NAME=speech_translation
MODEL_NAME=src-asr-like.asr-bpe.base.ft-mustc.lr00003
GROUP_NAME=mt.mustc.en-de

FAIRSEQ_PATH=/work/smt4/gao/tran/speech_translation/fairseq
DATA_PATH=/work/smt4/gao/tran/speech_translation/datasets/text/manual/parnia.mustc-mt.en-de.asr-bpe.src-normalized/data-bin.v2
MODEL_PATH=/work/smt4/gao/tran/$PROJECT_NAME/trained_models/$GROUP_NAME/$MODEL_NAME
USER_DIR=$FAIRSEQ_PATH/projects/$PROJECT_NAME


read -r -d '' EVAL_ARGS <<EOF
--max-tokens 8192 \
--path $MODEL_PATH/checkpoints/checkpoint_best.pt \
--remove-bpe sentencepiece \
--results-path $MODEL_PATH/results \
--beam 12 \
--nbest 1 \
--buffer-size 2000
EOF

read -r -d '' OTHER_ARGS <<EOF
--num-workers 1 \
--no-progress-bar \
--log-interval 50 \
--log-format simple \
--log-file $MODEL_PATH/generate.log \
--user-dir $USER_DIR
EOF

mkdir -p $MODEL_PATH
cd $MODEL_PATH

echo '-------------------------'
echo "Current fairseq commit: "
echo "$(git -C $FAIRSEQ_PATH rev-parse HEAD)"
echo '-------------------------'

# https://www-i6.informatik.rwth-aachen.de/publications/download/1139/Bahar-IWSLT-2020.pdf
# Describes training of transformer in section 2.4
echo 'Running fairseq-generate with following parameters...'
echo $DATA_PATH --task custom_translation $EVAL_ARGS $OTHER_ARGS
echo '-------------------------'

cat /work/smt4/gao/tran/speech_translation/datasets/text/manual/tstCOMMON.mustc-mt.en-de.asr-bpe/test.enc.en-de.en | \
  python /work/smt2/vtran/fairseq-st/fairseq_cli/interactive.py \
    $DATA_PATH --task custom_translation $EVAL_ARGS $OTHER_ARGS > $MODEL_PATH/gen_COMMON.txt

cat /work/smt4/gao/tran/speech_translation/datasets/text/manual/tstHE.mustc-mt.en-de.asr-bpe/test.enc.en-de.en | \
  python /work/smt2/vtran/fairseq-st/fairseq_cli/interactive.py \
    $DATA_PATH --task custom_translation $EVAL_ARGS $OTHER_ARGS > $MODEL_PATH/gen_HE.txt