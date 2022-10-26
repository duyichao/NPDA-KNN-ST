# NPDA-kNN-ST
Official implementation of EMNLP'2022 paper "Non-Parametric Domain Adaptation for End-to-end Speech Translation".

This codebase is currently a nightly version and is undergoing refactoring, and we will release the refactored code in the future.

## Citation

Please cite our paper if you find this repository helpful in your research:

```
@article{Du2022NonParametricDA,
  title={Non-Parametric Domain Adaptation for End-to-End Speech Translation},
  author={Yichao Du and Weizhi Wang and Zhirui Zhang and Boxing Chen and Tong Xu and Jun Xie and Enhong Chen},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.11211}
}
```

# Instructions

## Requirements and Insallation

* python = 3.6
* pytorch = 1.8.1
* torchaudio = 0.8.1
* SoundFile = 0.10.3.post1
* numpy = 1.19.5
* omegaconf = 2.0.6
* PyYAML = 5.4.1
* sentencepiece = 0.1.96
* sacrebleu = 1.5.1
* faiss-gpu = 1.7.1.post1
* torch-scatter = 2.0.8

## Preparations and Configurations
### Pre-trained Model and Data
We use the vocab file and pre-trained ST model provided by [Fairseq S2T MuST-C Example](https://github.com/pytorch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md). 

#### TSV Data

The TSV manifests we used are different from Fairseq S2T MuST-C Example, as follows:

``` tsv
id	audio	n_frames	speaker	src_lang	src_text	tgt_lang	tgt_text
ted_767_0	/data/mustc/en-fr/fbank80.zip:55688475685:274688	858	spk.767	en	These breakthroughs, we need to move those at full speed, and we can measure that in terms of companies, pilot projects, regulatory things that have been changed.	fr	Ces progrès, il faut que nous les réalisions à toute vitesse, et nous pouvons quantifier cela en termes de sociétés, de projets pilotes, de modifications des dispositions réglementaires.
```

You can get them by:
``` bash 
cd ./myscripts/prepare_data
bash ./prepare_data/prep_mustc_data.sh
bash ./prep_europarst_data.sh
bash ./prep_europarmt_data.sh
```

#### Config File

If the source and target dictionaries are different, we need to declare `src_bpe_tokenizer`, `src_vocab_filename`, `tgt_bpe_tokenizer`, and `vocab_filename` in config file. 

```yaml
input_channels: 1
input_feat_per_channel: 80
sampling_alpha: 1.0
specaugment:
  freq_mask_F: 27
  freq_mask_N: 1
  time_mask_N: 1
  time_mask_T: 100
  time_mask_p: 1.0
  time_wrap_W: 0
transforms:
  '*':
  - utterance_cmvn
  _train:
  - utterance_cmvn
  - specaugment
src_bpe_tokenizer:
  bpe: sentencepiece
  sentencepiece_model: /data/mustc/en-fr/st_joint_mt_data/spm_unigram_5000.model
src_vocab_filename: /data/mustc/en-fr/st_joint_mt_data/spm_unigram_5000.txt
tgt_bpe_tokenizer:
  bpe: sentencepiece
  sentencepiece_model: /data/mustc/en-fr/st_joint_mt_data/spm_unigram_8000.model
vocab_filename: /data/mustc/en-fr/st_joint_mt_data/spm_unigram_8000.txt
```

For multilingual experiments, you need to add a new parameter `prepend_tgt_lang_tag: True` to the configuration yaml file. 

## Unifying Text and Speech Representation

We input parallel $\langle speech, translation\rangle$  and $\langle speech, transcription\rangle$  into the model to unify text adn speech representation on the decoder side. For convenience, we also provide the well trained  [model](https://drive.google.com/drive/folders/1Ds394y26IifaBXEfA3g2rlrpne-o0qTd?usp=sharing).

Trianing Bilingual Model on MuST-C Corpus:
```bash
cd ./myscripts/mustc2europarl
bash ./unify_representation_bilingual.sh
```

Trianing Multilingual Model on MuST-C Corpus:
```bash
cd ./myscripts/mustc2europarl
bash ./unify_representation_multilingual.sh
```

## Inference with $k$NN Retrieval
### Create Datastore

When the model with unified text and speech representation is tuned well, we could load the model for creating a cached datastore with the script as follow:

```bash
cd ./myscripts/mustc2europarlst
bash ./build_datastore.sh
```

### Build Faiss Index

The FAISS index requires a training stage where it learns a set of clusters for the keys. Once this is completed, the keys must all be added to the index. The speed of adding keys to the index depends on the hardware, particularly the amount of RAM available. When the knn_index is build, we can remove `keys.npy` and `vals.key` to save the hard disk space. 

```bash
cd ./myscripts/mustc2europarlst
bash ./train_datastore.sh
```

### Inference via $k$NN-ST

```bash
cd ./myscripts/mustc2europarlst
bash ./eval_via_knn.sh
```

