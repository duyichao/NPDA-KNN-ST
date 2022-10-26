#!/usr/bin/env bash

TGT_LANG=$1
DS_TYPE=$2
CUDA_IDS=$3
temperature_value=$4
lambda_value=$5
k=$6

DSTORE_ROOT=/path/to/datastore
EUROPARL_ST_ROOT=/path/to/epst/data

ST_SAVE_DIR=/path/to/save/model
CHECKPOINT_FILENAME=/ckpt/name

NNST_SCRIPTS=/path/to/KNN-ST/NNST

declare -A dic
dic=(
  ["de_st"]=1220631 ["de_mt_only"]=74795371
  ["es_st"]=1160737 ["es_mt_only"]=76226723
  ["nl_st"]=1153677 ["nl_mt_only"]=76011171
  ["it_st"]=1139009 ["it_mt_only"]=75981836
  ["ro_st"]=1083567 ["ro_mt_only"]=15738321
  ["pt_st"]=1194161 ["pt_mt_only"]=78375866
  ["fr_st"]=1265862 ["fr_mt_only"]=83303733
)

if [ ${DS_TYPE} == "st_text" ]; then
  DS_PATH=${DSTORE_ROOT}/ds_hidden_mapping_ep_st_data_text
  key=${TGT_LANG}_st
elif [ ${DS_TYPE} == "st_speech" ]; then
  DS_PATH=${DSTORE_ROOT}/ds_hidden_mapping_ep_st_data_speech
  key=${TGT_LANG}_st
elif [ ${DS_TYPE} == "st_speech_text" ]; then
  DS_PATH=${DSTORE_ROOT}/ds_hidden_mapping_ep_st_data_speech_text
  key=${TGT_LANG}_st_speech_text
elif [ ${DS_TYPE} == "mt_text_only" ]; then
  DS_PATH=${DSTORE_ROOT}/ds_hidden_mapping_ep_mt_data_text_only
  key=${TGT_LANG}_mt_only
fi

DS_SIZE=${dic[$key]}

echo "TGT_LANG:${TGT_LANG}, DS_TYPE:${DS_TYPE}, DS_PATH:${DS_PATH}, DS_SIZE:${DS_SIZE}"

CUDA_VISIBLE_DEVICES=${CUDA_IDS} \
  python3 ${NNST_SCRIPTS}/generate_knn.py \
  ${EUROPARL_ST_ROOT}/${TGT_LANG} --config-yaml config_bilingual_unify_rep.yaml \
  --gen-subset dev_joint --task speech_to_text_joint_mt \
  --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --batch-size 64 --beam 5 --scoring sacrebleu \
  --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '${DS_PATH}',
'dstore_size': ${DS_SIZE}, 'dstore_fp16': True, 'k': ${k}, 'probe': 64, 'knn_sim_func': 'do_not_recomp_l2',
'use_gpu_to_search': True, 'move_dstore_to_mem': True, 'no_load_keys': True, 'knn_lambda_type': 'fix',
'knn_lambda_value': ${lambda_value}, 'knn_temperature_type': 'fix', 'knn_temperature_value': ${temperature_value},
'load_pretrained_st_model_from': '/path/to/mustc_de_st_transformer_s.pt'}" \
  --generate-task-type st --quiet
