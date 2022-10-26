#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
import shutil
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
    cal_gcmvn_stats,
)
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from fairseq.data.audio.audio_utils import get_waveform

from fairseq.data.audio.audio_utils import get_waveform

log = logging.getLogger(__name__)

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_lang", "speaker", "src_text", "tgt_lang", "tgt_text"]


class EuroparlST(Dataset):
    SPLITS = ["train", "dev", "test"]
    # SPLITS = ["dev"]
    SRC_LANGUAGES = ["en"]
    TGT_LANGUAGES = ["de", "fr", "es", "it", "nl", "pt", "ro"]

    def __init__(self, root: str, src_lang: str, tgt_lang: str, split: str) -> None:
        # TODO process speaker_id
        assert split in self.SPLITS and tgt_lang in self.TGT_LANGUAGES
        assert src_lang in self.SRC_LANGUAGES
        _root = Path(root) / f"{src_lang}"
        wav_root, txt_root = _root / "audios_wav", _root / f"{tgt_lang}" / split
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()

        segments = []

        log.info("reading segments.lst...")
        with open(txt_root / "segments.lst", 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split("\n")[0].split(" ")
                segments.append(
                    {
                        "wav": line[0] + ".wav",
                        "offset": line[1],
                        "duration": float(line[2]) - float(line[1])
                    }
                )
            f.close()

        log.info("reading segments.src and segments.tgt...")
        for _lang in [src_lang, tgt_lang]:
            with open(txt_root / f"segments.{_lang}", 'r') as f:
                utterances = [r.strip() for r in f]
                assert len(segments) == len(utterances)
            for idx, utt in enumerate(utterances):
                segments[idx][_lang] = utt

        self.data = []

        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: x["offset"])
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment[src_lang],
                        segment[tgt_lang],
                        _id,
                    )
                )
        # log.info(self.data[:10])

    def __getitem__(self, n: int):
        # TODO process speaker_id
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, utt_id = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
        waveform = torch.from_numpy(waveform)
        # return waveform, offset, n_frames, sr, src_utt, tgt_utt, utt_id
        return waveform, sr, src_utt, tgt_utt, utt_id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = Path(args.data_root).absolute()
    for tgt_lang in EuroparlST.TGT_LANGUAGES:
        log.info(f"start processing {tgt_lang}....")
        cur_root = root / "en" / f"{tgt_lang}"
        if not cur_root.is_dir():
            log.info(f"{cur_root.as_posix()} does not exist. Skipped.")
            continue
        feature_root = cur_root / "fbank80"
        feature_root.mkdir(exist_ok=True)
        
        for split in EuroparlST.SPLITS:
            log.info(f"Fetching split {split}...")
            dataset = EuroparlST(root.as_posix(), src_lang="en", tgt_lang=tgt_lang, split=split)
            log.info("Extracting log mel filter bank features...")
            if split == 'train' and args.cmvn_type == "global":
                log.info("And estimating cepstral mean and variance stats...")
                gcmvn_feature_list = []
        
            for waveform, sample_rate, _, _, utt_id in tqdm(dataset):
                # for waveform, offset, n_frames, sample_rate, src_utt, tgt_utt, utt_id in dataset:
                # print(utt_id, n_frames, offset, waveform)
                features = extract_fbank_features(waveform, sample_rate)
        
                np.save(
                    (feature_root / f"{utt_id}.npy").as_posix(),
                    features
                )
            if split == 'train' and args.cmvn_type == "global":
                if len(gcmvn_feature_list) < args.gcmvn_max_num:
                    gcmvn_feature_list.append(features)
        
        # Pack features into ZIP
        zip_path = cur_root / "fbank80.zip"
        log.info("ZIPing features...")
        create_zip(feature_root, zip_path)
        log.info("Fetching ZIP manifest...")
        zip_manifest = get_zip_manifest(zip_path)
        # Generate TSV manifest
        log.info("Generating manifest...")
        train_text = []
        for split in EuroparlST.SPLITS:
            is_train_split = split.startswith("train")
            manifest = {c: [] for c in MANIFEST_COLUMNS}
            dataset = EuroparlST(args.data_root, "en", tgt_lang, split)
            for wav, sr, src_utt, tgt_utt, utt_id in tqdm(dataset):
                manifest["id"].append(utt_id)
                manifest["speaker"].append("")
                manifest["audio"].append(zip_manifest[utt_id])
                duration_ms = int(wav.size(1) / sr * 1000)
                manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
                manifest["src_lang"].append("en" if args.task != "asr" else "")
                manifest["tgt_lang"].append(tgt_lang if args.task != "asr" else "en")
                manifest["src_text"].append(src_utt if args.task != "asr" else "")
                manifest["tgt_text"].append(src_utt if args.task == "asr" else tgt_utt)
            if is_train_split:
                if args.task != "joint":
                    train_text.extend(manifest["tgt_text"])
                else:
                    train_text.extend(manifest["tgt_text"])
                    train_text.extend(manifest["src_text"])
            df = pd.DataFrame.from_dict(manifest)
            df = filter_manifest_df(df, is_train_split=is_train_split)
            save_df_to_tsv(df, cur_root / f"{split}_{args.task}.tsv")

        # Generate vocab
        v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
        spm_filename_prefix = f"spm_{args.vocab_type}_{v_size_str}_{args.task}"
        with NamedTemporaryFile(mode="w") as f:
            for t in train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                cur_root / spm_filename_prefix,
                args.vocab_type,
                args.vocab_size,
            )
        # Generate config YAML
        gen_config_yaml(
            cur_root,
            spm_filename_prefix + ".model",
            yaml_filename=f"config_{args.task}.yaml",
            specaugment_policy="lb",
            cmvn_type=args.cmvn_type,
            gcmvn_path=(
                cur_root / "gcmvn.npz" if args.cmvn_type == "global"
                else None
            ),
        )
        # Clean up
        shutil.rmtree(feature_root)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
        default="unigram",
    ),
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("--task", type=str, choices=["asr", "st", "joint"])
    parser.add_argument("--joint", action="store_true", help="")
    parser.add_argument("--cmvn-type", default="utterance",
                        choices=["global", "utterance"],
                        help="The type of cepstral mean and variance normalization")
    parser.add_argument("--gcmvn-max-num", default=150000, type=int,
                        help=(
                            "Maximum number of sentences to use to estimate"
                            "global mean and variance"
                        ))
    args = parser.parse_args()

    process(args)
