#!/bin/bash

echo 'Activating venv....'
source /work/smt4/gao/tran/speech_translation/venv-fairseq-py39/bin/activate

PROJECT_NAME=speech_translation
MODEL_NAME=joint-seq.mustc.en-de.base.multitask.inversesqrt
GROUP_NAME=joint-seq.mustc

FAIRSEQ_PATH=/work/smt4/gao/tran/speech_translation/fairseq
MODEL_PATH=/work/smt4/gao/tran/$PROJECT_NAME/trained_models/$GROUP_NAME/$MODEL_NAME
USER_DIR=$FAIRSEQ_PATH/projects/$PROJECT_NAME

read -r -d '' EVAL_ARGS <<EOF
--data-asr /work/smt2/vtran/datasets/asr/parnia.mustc.asr.v2 \
--config-yaml-asr /work/smt2/vtran/datasets/asr/parnia.mustc.asr.v2/config_asr.i6approx.yaml \
--data-st /work/smt2/vtran/datasets/asr/parnia.mustc.st \
--config-yaml-st /work/smt2/vtran/datasets/asr/parnia.mustc.st/config_st.mt-bpe.yaml \
--data-mt /work/smt4/gao/tran/speech_translation/datasets/text/out/parnia.mustc-mt.en-de.32k.src-normalized.fix/data-bin \
--task cascaded_speech_translation \
--source-lang en \
--target-lang de \
--pivot-spm-model /work/smt4/gao/tran/speech_translation/datasets/text/out/bpe_vocab/parnia.iwslt2020_mt_en-de.lowercase-nopunct.32k.en.model \
--batch-size 1 \
--max-tokens 8000 \
--log-format simple \
--log-interval 100 \
--path $MODEL_PATH/checkpoints/checkpoint_best.pt \
--beam 12 \
--nbest 1 \
--max-source-positions 9999 \
--max-target-positions 9999 \
--ensemble-decoding
EOF

read -r -d '' OTHER_ARGS <<EOF
--num-workers 1 \
--no-progress-bar \
--log-file $MODEL_PATH/eval.log \
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
echo --task custom_speech_to_text $MODEL_ARGS $TRAIN_ARGS $OTHER_ARGS
echo '-------------------------'

python $FAIRSEQ_PATH/fairseq_cli/generate.py \
  --task speech_to_text --gen-subset tst-COMMON_st --results-path ensemble_results_COMMON $MODEL_ARGS $EVAL_ARGS $OTHER_ARGS

python $FAIRSEQ_PATH/fairseq_cli/generate.py \
  --task speech_to_text --gen-subset tst-HE_st --results-path ensemble_results_HE $MODEL_ARGS $EVAL_ARGS $OTHER_ARGS
