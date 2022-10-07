#!/bin/bash

echo 'Activating venv....'
source /work/smt4/gao/tran/speech_translation/venv-fairseq-py39/bin/activate

PROJECT_NAME=speech_translation
MODEL_NAME=$1
GROUP_NAME=joint.mustc.en-de.ft

FAIRSEQ_PATH=/work/smt4/gao/tran/speech_translation/fairseq
DATA_PATH=/work/smt2/vtran/datasets/asr/parnia.mustc.asr.v2
MODEL_PATH=/work/smt4/gao/tran/$PROJECT_NAME/trained_models/$GROUP_NAME/$MODEL_NAME
USER_DIR=$FAIRSEQ_PATH/projects/$PROJECT_NAME

read -r -d '' EVAL_ARGS <<EOF
--batch-size 200 \
--max-tokens 10000 \
--log-format simple \
--log-interval 100 \
--path $MODEL_PATH/checkpoints/checkpoint_best.asr.pt \
--beam 12 \
--nbest 1 \
--scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct \
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
echo $DATA_PATH --task custom_speech_to_text --config-yaml $DATA_PATH/config_asr.yaml $MODEL_ARGS $TRAIN_ARGS $OTHER_ARGS
echo '-------------------------'

echo "Extracting asr checkpoint..."
python $FAIRSEQ_PATH/scripts/extract_model_from_cascade.py $MODEL_PATH/checkpoints/checkpoint_best.pt asr
echo "Done."

cd $MODEL_PATH
python $FAIRSEQ_PATH/fairseq_cli/generate.py \
  $DATA_PATH --task speech_to_text --gen-subset tst-COMMON_asr --results-path asr_results_COMMON --config-yaml $DATA_PATH/config_asr.yaml $MODEL_ARGS $EVAL_ARGS $OTHER_ARGS
cd asr_results_COMMON
create_ctm.py generate-tst-COMMON_asr.txt out.ctm --stm /work/bahar/st/iwslt2020/data/final_data/ref/asr/tstCOMMON.stm
/work/bahar/st/iwslt2020/data/final_data/ref/asr/eval-sclite-wer.sh tstCOMMON out.ctm > final_wer

cd $MODEL_PATH
python $FAIRSEQ_PATH/fairseq_cli/generate.py \
  $DATA_PATH --task speech_to_text --gen-subset tst-HE_asr --results-path asr_results_HE --config-yaml $DATA_PATH/config_asr.yaml $MODEL_ARGS $EVAL_ARGS $OTHER_ARGS
cd asr_results_HE
create_ctm.py generate-tst-HE_asr.txt out.ctm --stm /work/bahar/st/iwslt2020/data/final_data/ref/asr/tstHE.stm
/work/bahar/st/iwslt2020/data/final_data/ref/asr/eval-sclite-wer.sh tstHE out.ctm > final_wer