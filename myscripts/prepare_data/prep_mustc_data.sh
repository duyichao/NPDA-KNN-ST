#!/usr/bin/env bash

# pre processing Europarl-ST data
ST_PATH=/path/to/mustc/data

python3 ./prep_europarlst_data.py \
  --data-root ${ST_PATH} --task asr \
  --vocab-type unigram --vocab-size 5000

python3 ./prep_europarlst_data.py \
  --data-root ${ST_PATH} --task st \
  --vocab-type unigram --vocab-size 8000