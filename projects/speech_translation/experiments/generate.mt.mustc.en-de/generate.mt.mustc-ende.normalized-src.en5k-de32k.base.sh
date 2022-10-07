#!/bin/bash

echo 'Activating venv....'
source /work/smt2/vtran/virtualenvs/fairseq-py39/bin/activate

PROJECT_NAME=speech_translation
MODEL_NAME=normalized-src.en5k-de32k.base
GROUP_NAME=mt.mustc-ende

DATA_PATH=/work/smt2/vtran/datasets/mt/out/parnia.mustc-mt.en-de.en-5k.de-32k.src-normalized/data-bin
MODEL_PATH=/work/smt2/vtran/projects/$PROJECT_NAME/$GROUP_NAME/$MODEL_NAME
USER_DIR=/work/smt2/vtran/fairseq-st/projects/speech_translation


read -r -d '' EVAL_ARGS <<EOF
--max-tokens 8192 \
--amp \
--path $MODEL_PATH/checkpoints/checkpoint_best.pt \
--remove-bpe sentencepiece \
--results-path $MODEL_PATH/generated \
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
echo "$(git -C /u/vtran/work/fairseq-st rev-parse HEAD)"
echo '-------------------------'

# https://www-i6.informatik.rwth-aachen.de/publications/download/1139/Bahar-IWSLT-2020.pdf
# Describes training of transformer in section 2.4
echo 'Running fairseq-generate with following parameters...'
echo $DATA_PATH --task custom_translation $EVAL_ARGS $OTHER_ARGS
echo '-------------------------'

cat /work/smt2/vtran/datasets/mt/out/mustc-mt.test-COMMON.en-de.en-5k.de-32k.src-normalized/test.enc.en-de.en | \
  python /work/smt2/vtran/fairseq-st/fairseq_cli/interactive.py \
    $DATA_PATH --task custom_translation $EVAL_ARGS $OTHER_ARGS > $MODEL_PATH/gen_COMMON.txt

cat /work/smt2/vtran/datasets/mt/out/mustc-mt.test-HE.en-de.en-5k.de-32k.src-normalized/test.enc.en-de.en | \
  python /work/smt2/vtran/fairseq-st/fairseq_cli/interactive.py \
    $DATA_PATH --task custom_translation $EVAL_ARGS $OTHER_ARGS > $MODEL_PATH/gen_HE.txt
