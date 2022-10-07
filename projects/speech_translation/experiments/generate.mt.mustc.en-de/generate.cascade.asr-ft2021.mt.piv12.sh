#!/bin/bash

echo 'Activating venv....'
source /work/smt4/gao/tran/speech_translation/venv-fairseq-py39/bin/activate

PROJECT_NAME=speech_translation
MODEL_NAME=src-asr-like.bpe-separate.base.fix
GROUP_NAME=mt.mustc.en-de

FAIRSEQ_PATH=/work/smt4/gao/tran/speech_translation/fairseq
DATA_PATH=/work/smt4/gao/tran/speech_translation/datasets/text/out/parnia.mustc-mt.en-de.32k.src-normalized.fix/data-bin
MODEL_PATH=/work/smt4/gao/tran/$PROJECT_NAME/trained_models/$GROUP_NAME/$MODEL_NAME
USER_DIR=$FAIRSEQ_PATH/projects/$PROJECT_NAME


read -r -d '' EVAL_ARGS <<EOF
--max-tokens 4096 \
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

mkdir -p $MODEL_PATH/results_CASCADED.asr-ft2021
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

ASR_COMMON=/work/smt4/gao/tran/speech_translation/trained_models/asr.mustc.en-de/fair.ctc.ft2021.lr1e-4.noctc/results_COMMON_12best/generate-tst-COMMON_asr.txt
ASR_HE=/work/smt4/gao/tran/speech_translation/trained_models/asr.mustc.en-de/fair.ctc.ft2021.lr1e-4.noctc/results_HE_12best/generate-tst-HE_asr.txt
RESULTS_PATH=$MODEL_PATH/results_CASCADED.asr-ft2021.piv12
mkdir -p $RESULTS_PATH


echo "Extracting ASR hypotheses..."
cat $ASR_COMMON | grep ^H- | sed s/H-//| sort -h | cut -f3 > $RESULTS_PATH/asr_COMMON.asr_enc.txt
cat $ASR_HE | grep ^H- | sed s/H-//| sort -h | cut -f3 > $RESULTS_PATH/asr_HE.asr_enc.txt

echo "spm_decode..."
python3 $USER_DIR/evaluation/spm_decode.py $RESULTS_PATH/asr_COMMON.asr_enc.txt > $RESULTS_PATH/asr_COMMON.txt
python3 $USER_DIR/evaluation/spm_decode.py $RESULTS_PATH/asr_HE.asr_enc.txt > $RESULTS_PATH/asr_HE.txt

echo "spm_encode to MT bpe..."
python3 $USER_DIR/data_prep/spm_encode.py /work/smt2/vtran/datasets/mt/out/bpe_vocab/parnia.iwslt2020_mt_en-de.lowercase-nopunct.32k.en.model \
  $RESULTS_PATH/asr_COMMON.txt > $RESULTS_PATH/asr_COMMON.mt_enc.txt
  python3 $USER_DIR/data_prep/spm_encode.py /work/smt2/vtran/datasets/mt/out/bpe_vocab/parnia.iwslt2020_mt_en-de.lowercase-nopunct.32k.en.model \
  $RESULTS_PATH/asr_HE.txt > $RESULTS_PATH/asr_HE.mt_enc.txt

echo "Start MT decoding..."
cat $RESULTS_PATH/asr_COMMON.mt_enc.txt | \
  python /work/smt2/vtran/fairseq-st/fairseq_cli/interactive.py \
    $DATA_PATH --task custom_translation $EVAL_ARGS $OTHER_ARGS > $RESULTS_PATH/gen_COMMON.txt

cat $RESULTS_PATH/asr_HE.mt_enc.txt | \
  python /work/smt2/vtran/fairseq-st/fairseq_cli/interactive.py \
    $DATA_PATH --task custom_translation $EVAL_ARGS $OTHER_ARGS > $RESULTS_PATH/gen_HE.txt
