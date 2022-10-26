#!/usr/bin/env bash

TGT_LANG=$1
DS_TYPE=$2
CUDA_IDS=$3

DSTORE_ROOT=
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
elif [ ${DS_TYPE} == "mt_text_only" ]; then
  DS_PATH=${DSTORE_ROOT}/ds_hidden_mapping_ep_mt_data_text_only
  key=${TGT_LANG}_mt_only
fi
DS_SIZE=${dic[$key]}
echo "TGT_LANG:${TGT_LANG}, DS_TYPE:${DS_TYPE}, DS_PATH:${DS_PATH}, DS_SIZE:${DS_SIZE}"

rm ${DS_PATH}/knn_index*

CUDA_VISIBLE_DEVICES=${CUDA_IDS} \
  python3 ${NNST_SCRIPTS}/train_datastore_gpu.py \
  --dstore_mmap ${DS_PATH} --dstore_size ${DS_SIZE} --dstore-fp16 \
  --faiss_index ${DS_PATH}/knn_index \
  --ncentroids 8192 --probe 64 --dimension 256 \
  --starting_point 0 --num_keys_to_add_at_a_time 50000
