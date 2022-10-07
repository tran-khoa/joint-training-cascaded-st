#!/usr/bin/env fish
set MTEDX_ROOT /backup/datasets/mtedx/

#python prep_mtedx_data.py --data-root /backup/datasets/mtedx/ --task asr --vocab-type unigram --vocab-size 1000
python prep_mtedx_data.py --data-root /backup/datasets/mtedx/ --task st --vocab-type unigram --vocab-size 1000
#python prep_mtedx_data.py \
#  --data-root ${MTEDX_ROOT} --task st \
#  --vocab-type unigram --vocab-size 1000
