#!/usr/bin/env/python

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os.path as op
from argparse import Namespace

import torch

from fairseq.data import Dictionary, encoders
from fairseq.data.audio.speech_to_text_joint_mt_dataset import (
    SpeechToTextJointMTDataset,
    SpeechToTextJointMTDatasetCreator,
    get_features_or_waveform,
    S2TJMTDataConfig,
)
from fairseq.tasks import LegacyFairseqTask, register_task

logger = logging.getLogger(__name__)


@register_task("speech_to_text_joint_mt")
class SpeechToTextJointMTTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--hidden-mapping-task-flag",
            # default=False,
            action="store_true",
            help="Whether to perform the Hidden Mapping task."
        )
        parser.add_argument(
            "--generate-task-type",
            default='st',
            choices=["st", "mt"],
            help="Choosing a task for inference."
        )

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TJMTDataConfig(op.join(args.data, args.config_yaml))

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TJMTDataConfig(op.join(args.data, args.config_yaml))
        # load target dict
        dict_path = op.join(args.data, data_cfg.vocab_filename)
        if not op.isfile(dict_path):
            raise FileNotFoundError(f"Target dict not found: {dict_path}")
        tgt_dict = Dictionary.load(dict_path)
        logger.info(
            f"target dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )
        # load source dict
        src_dict_path = op.join(args.data, data_cfg.src_vocab_filename)
        if not op.isfile(src_dict_path):
            raise FileNotFoundError(f"Source dict not found: {src_dict_path}")
        src_dict = Dictionary.load(src_dict_path)
        logger.info(
            f"source dictionary size ({data_cfg.src_vocab_filename}): " f"{len(src_dict):,}"
        )
        # # TODO train_subset限制太死，暂时注释了，后续需要修改一下
        # if getattr(args, "train_subset", None) is not None:
        #     if not all(s.startswith("train") for s in args.train_subset.split(",")):
        #         raise ValueError('Train splits should be named like "train*".')
        return cls(args, src_dict, tgt_dict)

    def build_criterion(self, args):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        src_bpe_tokenizer = self.build_src_bpe_tokenizer(self.args)
        assert src_bpe_tokenizer is not None
        tgt_bpe_tokenizer = self.build_tgt_bpe_tokenizer(self.args)
        assert tgt_bpe_tokenizer is not None
        joint_bpe_tokenizer = self.build_joint_bpe_tokenizer(self.args)
        self.datasets[split] = SpeechToTextJointMTDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            src_dict=self.src_dict,
            src_bpe_tokenizer=src_bpe_tokenizer,
            tgt_bpe_tokenizer=tgt_bpe_tokenizer,
            joint_bpe_tokenizer=joint_bpe_tokenizer,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return self.src_dict

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_model(self, args):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_channels
        return super(SpeechToTextJointMTTask, self).build_model(args)

    def build_generator(
            self,
            models,
            args,
            seq_gen_cls=None,
            extra_gen_cls_kwargs=None,
    ):
        # TODO Inference时会调用，此处可能需要修改
        if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if SpeechToTextJointMTDataset.is_lang_tag(s)
        }
        extra_gen_cls_kwargs = {"symbols_to_strip_from_output": lang_token_ids}
        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def inference_step(
            self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        # TODO 1. 需要统一speech,audio,st之类的命名规则 2. 现在访问mt_encoder的方式有些粗暴，需要结合s2t_jmt_transformer.py进行修改
        if self.args.generate_task_type == "st":
            sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"]["src_audios"]
            sample["net_input"]["src_lengths"] = sample["net_input"]["src_lengths"]["len_audios"]
            models[0].encoder = models[0].st_encoder
        elif self.args.generate_task_type == "mt":
            sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"]["src_texts"]
            sample["net_input"]["src_lengths"] = sample["net_input"]["src_lengths"]["len_texts"]
            models[0].encoder = models[0].mt_encoder

        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints
            )

    @staticmethod
    def forward_and_get_hidden_state_step(sample, model, modal="speech"):
        if modal == "speech":
            sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"]["src_audios"]
            sample["net_input"]["src_lengths"] = sample["net_input"]["src_lengths"]["len_audios"]
        elif modal == "text":
            sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"]["src_texts"]
            sample["net_input"]["src_lengths"] = sample["net_input"]["src_lengths"]["len_texts"]
        decoder_output, extra = model(
            src_tokens=sample['net_input']['src_tokens'],
            src_lengths=sample['net_input']['src_lengths'],
            prev_output_tokens=sample['net_input']['prev_output_tokens'],
            return_all_hiddens=False,
            features_only=True,
            modal=modal
        )
        return decoder_output

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    def build_src_bpe_tokenizer(self, args):
        logger.info(f"src-bpe-tokenizer: {self.data_cfg.src_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.src_bpe_tokenizer))

    def build_tgt_bpe_tokenizer(self, args):
        logger.info(f"tgt-bpe-tokenizer: {self.data_cfg.tgt_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.tgt_bpe_tokenizer))

    def build_joint_bpe_tokenizer(self, args):
        logger.info(f"joint-bpe-tokenizer: {self.data_cfg.joint_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.joint_bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        # TODO 要修改，现在是直接从s2t task中cp的；不影响训练
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        # TODO 要修改，现在是直接从s2t task中cp的；不影响训练
        return SpeechToTextJointMTDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )
