
CUDA_IDS=$1

PRETRAIN_ST_MODEL=/path/to/mustc_multilingual_st_transformer_m.pt
MUSTC_ROOT=/path/to/mustc/data
ST_SAVE_DIR=/path/to/save/model
mkdir $ST_SAVE_DIR -p

for lang in ru fr nl es it pt ro ; do 
    cp $MUSTC_ROOT/en-${lang}/st_joint_mt_data/train_joint.tsv $MUSTC_ROOT/multilingual/st_joint_mt_data/train_${lang}_joint.tsv
    cp $MUSTC_ROOT/en-${lang}/st_joint_mt_data/dev_joint.tsv $MUSTC_ROOT/multilingual/st_joint_mt_data/dev_${lang}_joint.tsv
    cp $MUSTC_ROOT/en-${lang}/st_joint_mt_data/tst-COMMON_joint.tsv $MUSTC_ROOT/multilingual/st_joint_mt_data/tst-COMMON_${lang}_joint.tsv
done

CUDA_VISIBLE_DEVICES=${CUDA_IDS} \
fairseq-train ${MUSTC_ROOT}/multilingual/st_joint_mt_data \
  --config-yaml config_multilingual_unify_rep.yaml \
  --train-subset train_de_joint,train_nl_joint,train_es_joint,train_fr_joint,train_it_joint,train_pt_joint,train_ro_joint,train_ru_joint \
  --valid-subset dev_de_joint,dev_nl_joint,dev_es_joint,dev_fr_joint,dev_it_joint,dev_pt_joint,dev_ro_joint,dev_ru_joint \
  --save-dir ${ST_SAVE_DIR} --num-workers 8 --max-tokens 20000 --max-update 300000 \
  --label-smoothing 0.1 --report-accuracy --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --arch s2t_jmt_transformer_s --task speech_to_text_joint_mt  \
  --criterion label_smoothed_cross_entropy_with_hidden_mapping \
  --pretrained-st-flag --forzen-st-model-params \
  --load-pretrained-st-model-from ${PRETRAIN_ST_MODEL} \
  --hidden-mapping-task-flag --hidden-mapping-loss-type mse \
  --occupy-gpu-flag --occupy-gpu-ids ${CUDA_IDS} \
  --ignore-prefix-size 1 --skip-invalid-size-inputs-valid-test \
  > $ST_SAVE_DIR/st-mt-hidden-mapping.log 2>&1