#!/usr/bin/env/python

import os
import sys
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("clean_europarl_data")

src_lang = "en"
tgt_langs = ["de", "fr", "es", "nl", "pt", "it", "ro"]

europarl_mt_path = "/data/europarl/v7"
europarl_st_path = "/data/s2t/europarl-st/v1.1/en"

for tgt_lang in tgt_langs:
    logger.info("process " + tgt_lang + " of Europarl-ST data....")

    # dev set
    dev_europarl = []
    with open("{}/{}/dev/speeches.{}".format(europarl_st_path, tgt_lang, tgt_lang)) as f:
        for idx, line in enumerate(f.readlines()):
            dev_europarl.append(line.strip("\n"))
    dev_europarl = set(dev_europarl)

    # test set
    test_europarl = []
    with open("{}/{}/test/speeches.{}".format(europarl_st_path, tgt_lang, tgt_lang)) as f:
        for idx, line in enumerate(f.readlines()):
            test_europarl.append(line.strip("\n"))
    test_europarl = set(test_europarl)

    # filted data
    europarl_clean_txt_path = "{}/filted_by_st/txt".format(
        europarl_mt_path)
    europarl_clean_tsv_path = "{}/filted_by_st/tsv".format(
        europarl_mt_path)
    if not os.path.exists(europarl_clean_tsv_path):
        os.makedirs(europarl_clean_tsv_path)
    if not os.path.exists(europarl_clean_txt_path):
        os.makedirs(europarl_clean_txt_path)


    lang_pair = tgt_lang + "-" + src_lang
    src_file = open("{}/europarl-v7.{}.{}".format(europarl_clean_txt_path, lang_pair, src_lang), "w")
    tgt_file = open("{}/europarl-v7.{}.{}".format(europarl_clean_txt_path, lang_pair, tgt_lang), "w")
    tsv_file = open("{}/train_epmt_joint.tsv".format(europarl_clean_tsv_path, tgt_lang), "w")
    tsv_file.write("id\taudio\tn_frames\tsrc_lang\tspeaker\tsrc_text\ttgt_lang\ttgt_text\n")

    with open("{}/raw/europarl-v7.{}.{}".format(europarl_mt_path, lang_pair, src_lang), "r") as sf:
        with open("{}/raw/europarl-v7.{}.{}".format(europarl_mt_path, lang_pair, tgt_lang), "r") as tf:
            for idx, (x, y) in enumerate(zip(sf.readlines(), tf.readlines())):
                x = x.strip("\n")
                y = y.strip("\n")
                if x == "" or y == "":
                    continue
                if idx % 1000000 == 0:
                    logger.info(tgt_lang + " " + str(idx) + " ok....")
                flag = 0
                for sent_t in test_europarl:
                    if y in sent_t and len(y.split()) >= 2:
                        flag = 1
                    if flag:
                        break
                if flag:
                    logger.info("source:" + x + "\t target:" + y)
                    continue
                for sent_d in dev_europarl:
                    if y in sent_d and len(y.split()) >= 2:
                        flag = 1
                    if flag:
                        break
                if flag:
                    logger.info("source:" + x + "\t target:" + y)
                    continue
                else:
                    src_file.write(x + "\n")
                    tgt_file.write(y + '\n')
                    
                    # # id	audio	n_frames	src_lang	speaker	src_text	tgt_lang	tgt_text
                    line = str(idx) + \
                           "\t/europarl-st/v1.1/en/de/fbank80.zip:5461633754:46208\t144" \
                           "\t{}\t\t{}\t{}\t{}\n".format(src_lang, x, tgt_lang, y)
                    tsv_file.write(line)

    logger.info("clean " + tgt_lang + " done....\n\n\n")
logger.info("clean all  done....\n\n\n")

if __name__ == '__main__':
    pass
