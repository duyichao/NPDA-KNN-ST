#!/usr/bin/env bash

SCRIPTS=/toolkit/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=/toolkit/subword-nmt/subword_nmt
BPE_TOKENS=40000



EP_MT_ROOT=path/to/europarl/v7/filted_by_st

python3 ./prep_europarlmt_data.py

for src in "en"; do
    for tgt in "de" "fr" "es" "nl" "ro" "pt" "it" ; do
        lang=$src-$tgt

        echo "######################################### preprocess ${src}-${tgt} #########################################"
        EP_MT_TXT=${EP_MT_ROOT}/txt
        EP_MT_DATA=${EP_MT_ROOT}/${src}-${tgt}
        MUSTC_DATA=/path/to/mustc/${src}-${tgt}/data
        prep=${EP_MT_DATA}/prep
        tmp=${EP_MT_DATA}/tmp
        rm -r ${prep} ${tmp}
        mkdir ${prep} ${tmp} -p

        echo "pre-processing ep data..."
        for l in $src $tgt; do
            rm $tmp/epmt.tags.$lang.tok.$l
            cat $EP_MT_TXT/europarl-v7.$tgt-$src.$l |
                perl $NORM_PUNC $l |
                perl $REM_NON_PRINT_CHAR |
                perl $TOKENIZER -threads 64 -a -l $l >$tmp/epmt.tags.$lang.tok.$l
        done

        echo "pre-processing mc data..."
        for l in $src $tgt; do
            rm $tmp/mustc.train.tags.$lang.tok.$l
            cat ${MUSTC_DATA}/train/txt/train.${l} |
                perl $NORM_PUNC $l |
                perl $REM_NON_PRINT_CHAR |
                perl $TOKENIZER -threads 64 -a -l $l >$tmp/mustc.train.tags.$lang.tok.$l
        done

        echo "splitting train and valid..."
        for l in $src $tgt; do
            awk '{if (NR%100 == 0)  print $0; }' $tmp/epmt.tags.$lang.tok.$l >$tmp/valid.$l
            awk '{if (NR%100 != 0)  print $0; }' $tmp/epmt.tags.$lang.tok.$l >$tmp/train.$l
        done

        TRAIN=${tmp}/train_temp
        BPE_CODE=$prep/code
        rm $TRAIN $BPE_CODE

        for l in $src $tgt; do
            cat $tmp/mustc.train.tags.$lang.tok.$l >>$TRAIN
        done

        echo "learn_bpe.py on ${TRAIN}..."
        python3 $BPEROOT/learn_bpe.py -s $BPE_TOKENS <$TRAIN >$BPE_CODE

        for L in $src $tgt; do
            for f in train.$L valid.$L; do
                echo "apply_bpe.py to ${f}..."
                python3 $BPEROOT/apply_bpe.py -c $BPE_CODE <$tmp/$f >$tmp/bpe.$f
            done
        done

        perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
        perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

        rm -r $EP_MT_DATA/data-bin/epmt_joint_mustc_dict_based
        mkdir $EP_MT_DATA/data-bin/epmt_joint_mustc_dict_based -p
        fairseq-preprocess \
            --source-lang $src --target-lang $tgt \
            --trainpref $prep/train \
            --validpref $prep/valid \
            --destdir $EP_MT_DATA/data-bin/epmt_joint_mustc_dict_based \
            --workers 64
        echo "######################################### preprocess ${src}-${tgt} done #########################################"
        echo ""
        echo ""
    done
done
