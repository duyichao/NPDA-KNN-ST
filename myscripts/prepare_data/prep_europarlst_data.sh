#!/usr/bin/env bash

# pre processing Europarl-ST data
ST_PATH=
M4A_PATH=${ST_PATH}/en/audios
WAV_PATH=${ST_PATH}/en/audios_wav
rm -rf $WAV_PATH
mkdir $WAV_PATH

src_langs=("en")
tgt_langs=("en" "de" "fr" "es" "ro" "it" "pt" "nl")

cd $M4A_PATH
for i in $(find . -type f); do
  ext="${i##*.}"
  if [[ $ext = m4a ]]; then
    p=${i%".m4a"}
    echo $i
    ffmpeg -v 0 -i $i -ac 1 -ar 16000 ${WAV_PATH}/$p'.wav'
  fi
done

python3 ./prep_europarlst_data.py \
  --data-root ${ST_PATH} --task asr \
  --vocab-type unigram --vocab-size 5000

python3 ./prep_europarlst_data.py \
  --data-root ${ST_PATH} --task st \
  --vocab-type unigram --vocab-size 8000