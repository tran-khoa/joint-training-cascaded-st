#!/bin/bash

echo 'Activating venv....'
source /work/smt4/gao/tran/speech_translation/venv-fairseq-py39/bin/activate

PROJECT_NAME=speech_translation
MODEL_NAME=untrained.asr-bpe
GROUP_NAME=joint-seq.mustc

FAIRSEQ_PATH=/work/smt4/gao/tran/speech_translation/fairseq
MODEL_PATH=/work/smt4/gao/tran/$PROJECT_NAME/trained_models/$GROUP_NAME/$MODEL_NAME
USER_DIR=$FAIRSEQ_PATH/projects/$PROJECT_NAME

read -r -d '' EVAL_ARGS <<EOF
--data-asr /work/smt2/vtran/datasets/asr/parnia.mustc.asr.v2 \
--config-yaml-asr /work/smt2/vtran/datasets/asr/parnia.mustc.asr.v2/config_asr.i6approx.yaml \
--data-st /work/smt2/vtran/datasets/asr/parnia.mustc.st \
--config-yaml-st /work/smt2/vtran/datasets/asr/parnia.mustc.st/config_st.mt-bpe.yaml \
--data-mt /work/smt4/gao/tran/speech_translation/datasets/text/manual/parnia.mustc-mt.en-de.asr-bpe.src-normalized/data-bin.v2 \
--task cascaded_speech_translation \
--source-lang en \
--target-lang de \
--batch-size 4 \
--pivot-beam 4 \
--max-tokens 8000 \
--log-format simple \
--log-interval 100 \
--path $MODEL_PATH/checkpoints/checkpoint_best.pt \
--beam 12 \
--nbest 1 \
--max-source-positions 9999 \
--max-target-positions 9999
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
echo --task custom_speech_to_text $MODEL_ARGS $EVAL_ARGS $OTHER_ARGS
echo '-------------------------'

for temp in 1.5 2 1 5 10
do
  echo "Evaluating for temperature $temp"

  python $FAIRSEQ_PATH/fairseq_cli/generate.py \
    --task speech_to_text --gen-subset tst-COMMON_st --results-path piv_temp${temp}_results_COMMON $MODEL_ARGS $EVAL_ARGS $OTHER_ARGS \
    --model-overrides "{'asr_model_conf':'$USER_DIR/experiments/train.joint-seq.mustc.en-de/asr.base.ft.yaml','mt_model_conf':'$USER_DIR/experiments/train.joint-seq.mustc.en-de/mt.base.yaml','tight_integrated':True,'tight_integrated_decoding_temp':$temp}" \

  cd $MODEL_PATH/piv_temp${temp}_results_COMMON
  /u/vtran/bin/eval_COMMON_de generate-tst-COMMON_st.txt
  cd $MODEL_PATH

  python $FAIRSEQ_PATH/fairseq_cli/generate.py \
    --task speech_to_text --gen-subset tst-HE_st --results-path piv_temp${temp}_results_HE $MODEL_ARGS $EVAL_ARGS $OTHER_ARGS \
    --model-overrides "{'asr_model_conf':'$USER_DIR/experiments/train.joint-seq.mustc.en-de/asr.base.ft.yaml','mt_model_conf':'$USER_DIR/experiments/train.joint-seq.mustc.en-de/mt.base.yaml','tight_integrated':True,'tight_integrated_decoding_temp':$temp}" \

  cd $MODEL_PATH/piv_temp${temp}_results_HE
  /u/vtran/bin/eval_HE_de generate-tst-HE_st.txt
  cd $MODEL_PATH
done