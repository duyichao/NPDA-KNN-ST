TGT_LANG=$1
CUDA_IDS=$2

MUSTC_ROOT=/path/to/mustc/data
PRETRAIN_ST_MODEL=/path/to/mustc_st_transformer_s.pt # download from https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md
ST_SAVE_DIR=/path/to/save/model
mkdir $ST_SAVE_DIR -p

CUDA_VISIBLE_DEVICES=${CUDA_IDS} \
  fairseq-train ${MUSTC_ROOT}/en-${TGT_LANG}/st_joint_mt_data \
  --config-yaml config_bilingual_unify_rep.yaml --train-subset train_joint --valid-subset dev_joint \
  --save-dir ${ST_SAVE_DIR} --num-workers 8 --max-tokens 20000 --max-update 500000 \
  --label-smoothing 0.1 --report-accuracy --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --arch s2t_jmt_transformer_s --task speech_to_text_joint_mt \
  --criterion label_smoothed_cross_entropy_with_hidden_mapping \
  --pretrained-st-flag --forzen-st-model-params \
  --load-pretrained-st-model-from ${PRETRAIN_ST_MODEL} \
  --hidden-mapping-task-flag --hidden-mapping-loss-type mse \
  --skip-invalid-size-inputs-valid-test
