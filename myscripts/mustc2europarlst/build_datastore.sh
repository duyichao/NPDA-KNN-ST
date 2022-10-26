#!/usr/bin/env bash

TGT_LANG=$1
DS_TYPE=$2
CUDA_IDS=$3

DSTORE_ROOT=
EUROPARL_ST_ROOT=
ST_SAVE_DIR=
CHECKPOINT_FILENAME=
NNST_SCRIPTS=/KNN-ST/NNST

VALID_SET="train_joint"
if [ ${DS_TYPE} == "st_text" ]; then
    # Europarl-ST speech data
  DS_PATH=${DSTORE_ROOT}/ds_hidden_mapping_ep_st_data_text
  GEN_TYPE="mt"
  DS_SIZE=1250000
elif [ ${DS_TYPE} == "st_speech" ]; then
  # Europarl-ST text data
  DS_PATH=${DSTORE_ROOT}/ds_hidden_mapping_ep_st_data_speech
  GEN_TYPE="st"
  DS_SIZE=1250000
elif [ ${DS_TYPE} == "mt_text_only" ]; then
  # only Europarl-MT data
  DS_PATH=${DSTORE_ROOT}/ds_hidden_mapping_ep_mt_data_text_only
  GEN_TYPE="mt"
  VALID_SET="train_epmt_joint"
  DS_SIZE=75000000
fi
echo "TGT_LANG:${TGT_LANG}, DS_TYPE:${DS_TYPE}, DS_PATH:${DS_PATH}"

rm -rf ${DS_PATH}
mkdir "${DS_PATH}" -p

CUDA_VISIBLE_DEVICES=${CUDA_IDS} \
python3 ${NNST_SCRIPTS}/save_datastore.py \
  ${EUROPARL_ST_ROOT}/${TGT_LANG} --config-yaml config_bilingual_unify_rep.yaml \
  --task speech_to_text_joint_mt --valid-subset ${VALID_SET} --dataset-impl mmap \
  --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --decoder-embed-dim 256 --dstore-fp16 --dstore-size ${DS_SIZE} \
  --dstore-mmap ${DS_PATH} \
  --skip-invalid-size-inputs-valid-test \
  --generate-task-type ${GEN_TYPE} \
  --model-overrides "{'load_pretrained_st_model_from': '/path/to/pretrained/st_model'}"




