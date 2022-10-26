#!/usr/bin/env/python


import logging
import math
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture, FairseqMultiModel, BaseFairseqModel, FairseqDecoder,
)
from fairseq.models.fairseq_model import check_type
from fairseq.models.transformer import Embedding, TransformerDecoder, base_architecture
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)
from torch import Tensor

from fairseq.models.speech_to_text.s2t_transformer import (
    Conv1dSubsampler,
    TransformerDecoderScriptable,
    S2TTransformerEncoder
)

logger = logging.getLogger(__name__)


class S2TJMTTransformerTextEncoder(FairseqEncoder):

    def __init__(self, args, src_dict, embed_tokens):
        super().__init__(src_dict)

        self.embed_tokens = embed_tokens

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )

        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

    def forward_embedding(self, src_tokens, token_embedding=None):
        # embed tokens and positions
        token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        # if self.layernorm_embedding is not None:
        #     x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        # if self.quant_noise is not None:
        #     x = self.quant_noise(x)
        return x, embed

    def _forward(self, src_tokens, src_lengths):
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = (src_tokens.device.type == "xla" or encoder_padding_mask.any())

        x, encoder_embedding = self.forward_embedding(src_tokens)

        # account for padding while computing the representation
        if encoder_padding_mask is not None:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        for layer in self.transformer_layers:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def forward(self, src_tokens, src_lengths):
        x = self._forward(src_tokens, src_lengths)
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            [] if len(encoder_out["encoder_padding_mask"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]]
        )

        new_encoder_embedding = (
            [] if len(encoder_out["encoder_embedding"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }


class S2TJMTTransformerSharedEncoder(FairseqEncoder):
    """Speech-to-text joint mt Transformer encoder that consists of input sub-sampler, token embedding and
    Transformer encoder."""

    def __init__(self, args, src_dict, embed_tokens):
        super().__init__(src_dict)

        self.encoder_freezing_updates = 0
        # self.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
        self.num_updates = 0

        self.embed_tokens = embed_tokens

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        self.subsample = Conv1dSubsampler(
            args.input_feat_per_channel * args.input_channels,
            args.conv_channels,
            args.encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
        )

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )

        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

    def forward_audio(self, src_tokens, src_lengths):
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = self.dropout_module(x)

        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask.any() else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def forward_embedding(self, src_tokens, token_embedding=None):
        # embed tokens and positions
        token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        # if self.layernorm_embedding is not None:
        #     x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        # if self.quant_noise is not None:
        #     x = self.quant_noise(x)
        return x, embed

    def forward_text(self, src_tokens, src_lengths):
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = (src_tokens.device.type == "xla" or encoder_padding_mask.any())

        x, encoder_embedding = self.forward_embedding(src_tokens)

        # account for padding while computing the representation
        if encoder_padding_mask is not None:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        for layer in self.transformer_layers:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def forward(self, src_tokens, src_lengths, modal):
        _forward = self.forward_audio if modal == "speech" else self.forward_text
        if self.num_updates < self.encoder_freezing_updates:
            with torch.no_grad():
                x = _forward(src_tokens, src_lengths)
        else:
            x = _forward(src_tokens, src_lengths)
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            [] if len(encoder_out["encoder_padding_mask"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]]
        )

        new_encoder_embedding = (
            [] if len(encoder_out["encoder_embedding"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }


class TransformerDecoderScriptable(TransformerDecoder):
    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, _ = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        last_hidden = x
        return x, last_hidden


# @register_model("s2t_jmt_transformer")
# class S2TJMTTransformerModel(FairseqEncoderDecoderModel):
#     """
#     Speech-to-text joint mt.
#     """
#
#     def __init__(self, encoder, decoder):
#         super().__init__(encoder, decoder)
#
#     @staticmethod
#     def add_args(parser):
#         """Add model-specific arguments to the parser."""
#         # input
#         parser.add_argument(
#             "--conv-kernel-sizes",
#             type=str,
#             metavar="N",
#             help="kernel sizes of Conv1d subsampling layers",
#         )
#         parser.add_argument(
#             "--conv-channels",
#             type=int,
#             metavar="N",
#             help="# of channels in Conv1d subsampling layers",
#         )
#         # Transformer
#         parser.add_argument(
#             "--activation-fn",
#             type=str,
#             default="relu",
#             choices=utils.get_available_activation_fns(),
#             help="activation function to use",
#         )
#         parser.add_argument(
#             "--dropout", type=float, metavar="D", help="dropout probability"
#         )
#         parser.add_argument(
#             "--attention-dropout",
#             type=float,
#             metavar="D",
#             help="dropout probability for attention weights",
#         )
#         parser.add_argument(
#             "--activation-dropout",
#             "--relu-dropout",
#             type=float,
#             metavar="D",
#             help="dropout probability after activation in FFN.",
#         )
#         parser.add_argument(
#             "--encoder-embed-dim",
#             type=int,
#             metavar="N",
#             help="encoder embedding dimension",
#         )
#         parser.add_argument(
#             "--encoder-ffn-embed-dim",
#             type=int,
#             metavar="N",
#             help="encoder embedding dimension for FFN",
#         )
#         parser.add_argument(
#             "--encoder-layers", type=int, metavar="N", help="num encoder layers"
#         )
#         parser.add_argument(
#             "--encoder-attention-heads",
#             type=int,
#             metavar="N",
#             help="num encoder attention heads",
#         )
#         parser.add_argument(
#             "--encoder-normalize-before",
#             action="store_true",
#             help="apply layernorm before each encoder block",
#         )
#         parser.add_argument(
#             "--decoder-embed-dim",
#             type=int,
#             metavar="N",
#             help="decoder embedding dimension",
#         )
#         parser.add_argument(
#             "--decoder-ffn-embed-dim",
#             type=int,
#             metavar="N",
#             help="decoder embedding dimension for FFN",
#         )
#         parser.add_argument(
#             "--decoder-layers", type=int, metavar="N", help="num decoder layers"
#         )
#         parser.add_argument(
#             "--decoder-attention-heads",
#             type=int,
#             metavar="N",
#             help="num decoder attention heads",
#         )
#         parser.add_argument(
#             "--decoder-normalize-before",
#             action="store_true",
#             help="apply layernorm before each decoder block",
#         )
#         parser.add_argument(
#             "--share-decoder-input-output-embed",
#             action="store_true",
#             help="share decoder input and output embeddings",
#         )
#         parser.add_argument(
#             "--layernorm-embedding",
#             action="store_true",
#             help="add layernorm to embedding",
#         )
#         parser.add_argument(
#             "--no-scale-embedding",
#             action="store_true",
#             help="if True, dont scale embeddings",
#         )
#         parser.add_argument(
#             "--load-pretrained-encoder-from",
#             type=str,
#             metavar="STR",
#             help="model to take encoder weights from (for initialization)",
#         )
#         parser.add_argument(
#             "--load-pretrained-decoder-from",
#             type=str,
#             metavar="STR",
#             help="model to take decoder weights from (for initialization)",
#         )
#         parser.add_argument(
#             '--encoder-freezing-updates',
#             default=None,
#             type=int,
#             metavar='N',
#             help='freeze encoder for first N updates'
#         )
#
#     @classmethod
#     def build_encoder(cls, args, src_dict, embed_tokens):
#         encoder = S2TJMTTransformerSharedEncoder(args, src_dict, embed_tokens)
#         pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
#         if pretraining_path is not None:
#             if not Path(pretraining_path).exists():
#                 logger.warning(
#                     f"skipped pretraining because {pretraining_path} does not exist"
#                 )
#             else:
#                 encoder = checkpoint_utils.load_pretrained_component_from_model(
#                     component=encoder, checkpoint=pretraining_path
#                 )
#                 logger.info(f"loaded pretrained encoder from: {pretraining_path}")
#         return encoder
#
#     @classmethod
#     def build_decoder(cls, args, tgt_dict, embed_tokens):
#         decoder = TransformerDecoderScriptable(args, tgt_dict, embed_tokens)
#         pretraining_path = getattr(args, "load_pretrained_decoder_from", None)
#         if pretraining_path is not None:
#             if not Path(pretraining_path).exists():
#                 logger.warning(
#                     f"skipped pretraining because {pretraining_path} does not exist"
#                 )
#             else:
#                 decoder = checkpoint_utils.load_pretrained_component_from_model(
#                     component=decoder, checkpoint=pretraining_path
#                 )
#                 logger.info(f"loaded pretrained decoder from: {pretraining_path}")
#         return decoder
#
#     @classmethod
#     def build_model(cls, args, task):
#         """Build a new model instance."""
#
#         # make sure all arguments are present in older models
#         base_s2t_jmt_architecture(args)
#
#         def build_embedding(dictionary, embed_dim):
#             num_embeddings = len(dictionary)
#             padding_idx = dictionary.pad()
#             return Embedding(num_embeddings, embed_dim, padding_idx)
#
#         # TODO 此处需要重写
#         encoder_embed_tokens = build_embedding(
#             task.src_dict, args.encoder_embed_dim
#         )
#
#         decoder_embed_tokens = build_embedding(
#             task.tgt_dict, args.decoder_embed_dim
#         )
#         encoder = cls.build_encoder(args, task.src_dict, encoder_embed_tokens)
#         decoder = cls.build_decoder(args, task.tgt_dict, decoder_embed_tokens)
#         return cls(encoder, decoder)
#
#     def get_normalized_probs(
#             self,
#             net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
#             log_probs: bool,
#             sample: Optional[Dict[str, Tensor]] = None,
#     ):
#         # net_output['encoder_out'] is a (B, T, D) tensor
#         lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
#         lprobs.batch_first = True
#         return lprobs
#
#     def forward(self, src_tokens, src_lengths, prev_output_tokens, modal="speech"):
#         """
#         The forward method inherited from the base class has a **kwargs
#         argument in its input, which is not supported in torchscript. This
#         method overwrites the forward method definition without **kwargs.
#         """
#         encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths, modal=modal)
#         decoder_out = self.decoder(
#             prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
#         )
#         return decoder_out


@register_model("s2t_jmt_transformer")
class S2TJMTTransformerModel(FairseqEncoderDecoderModel):
    """
    Speech-to-text joint mt.
    """

    def __init__(self, st_encoder, st_decoder, mt_encoder):
        super().__init__(st_encoder, st_decoder)
        self.st_encoder = self.encoder
        self.st_decoder = self.decoder
        self.mt_encoder = mt_encoder

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # input
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="N",
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d subsampling layers",
        )
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
        )
        parser.add_argument(
            '--load-pretrained-st-model-from',
            metavar='DIR',
            help='path to load checkpoint from pretrained mt model'
        )
        parser.add_argument(
            '--pretrained-st-flag',
            default=False,
            action='store_true',
            help='whether using pretrained st model'
        )
        parser.add_argument(
            '--encoder-freezing-updates',
            default=None,
            type=int,
            metavar='N',
            help='freeze encoder for first N updates'
        )
        parser.add_argument(
            "--share-encoder-decoder-embeddings",
            default=False,
            action="store_true",
            help="share decoder embeddings across languages",
        )
        parser.add_argument(
            "--share-encoders",
            action="store_true",
            help="share encoders across languages",
        )
        parser.add_argument(
            "--share-decoders",
            action="store_true",
            help="share decoders across languages",
        )
        parser.add_argument(
            "--freeze-st-model-params",
            action="store_true",
            help="freeze pretrained st model's parameters",
        )
        

    @classmethod
    def build_speech_encoder(cls, args, src_dict, embed_tokens):
        encoder = S2TTransformerEncoder(args)
        pretraining_path = getattr(args, "load_pretrained_speech_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained speech encoder from: {pretraining_path}")
        return encoder

    @classmethod
    def build_text_encoder(cls, args, src_dict, embed_tokens):
        encoder = S2TJMTTransformerTextEncoder(args, src_dict, embed_tokens)
        pretraining_path = getattr(args, "load_pretrained_text_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = TransformerDecoderScriptable(args, tgt_dict, embed_tokens)
        pretraining_path = getattr(args, "load_pretrained_decoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                decoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=decoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained decoder from: {pretraining_path}")
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_s2t_jmt_architecture(args)
        st_encoder, st_decoder = None, None
        if args.pretrained_st_flag:
            if args.use_knn_datastore:
                overrides = {
                    'load_knn_datastore': args.load_knn_datastore, 'use_knn_datastore': args.use_knn_datastore,
                    'dstore_filename': args.dstore_filename, 'dstore_size': args.dstore_size,
                    'dstore_fp16': args.dstore_fp16, 'k': args.k, 'knn_k_type': args.knn_k_type, 'probe': args.probe,
                    'knn_sim_func': args.knn_sim_func, 'use_gpu_to_search': args.use_gpu_to_search,
                    'move_dstore_to_mem': args.move_dstore_to_mem, 'no_load_keys': args.no_load_keys,
                    'knn_lambda_type': args.knn_lambda_type, 'knn_lambda_value': args.knn_lambda_value,
                    'knn_temperature_type': args.knn_temperature_type,
                    'knn_temperature_value': args.knn_temperature_value,
                    'faiss_metric_type': args.faiss_metric_type, 'max_k': args.max_k,
                }
                logger.info(overrides)
            else:
                overrides = None
                # exit(0)

            logger.info("loading pretrained st model...")
            if os.path.exists(args.load_pretrained_st_model_from):
                logger.info(f"pretrained st model path: {args.load_pretrained_st_model_from}")
                pretrained_st_model = checkpoint_utils.load_model_ensemble(
                    filenames=[args.load_pretrained_st_model_from],
                    arg_overrides=overrides,
                    task=task,
                )[0][0]
                # print(list(pretrained_st_model.children()))
                # exit(0)
                st_encoder = list(pretrained_st_model.children())[0]
                st_decoder = list(pretrained_st_model.children())[1]
            else:
                logger.info("pretrained st model path error.")
                raise RuntimeError

            # freeze pretrained model
            freeze_st_model_params_flag = getattr(args, "freeze_st_model_params", None)
            if freeze_st_model_params_flag:
                for param in st_encoder.parameters():
                    param.requires_grad = False
                for param in st_decoder.parameters():
                    param.requires_grad = False

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        if args.share_encoder_decoder_embeddings:
            encoder_embed_tokens = decoder_embed_tokens = build_embedding(
                task.tgt_dict, args.decoder_embed_dim
            )
        else:
            encoder_embed_tokens = build_embedding(
                task.src_dict, args.encoder_embed_dim
            )
            decoder_embed_tokens = build_embedding(
                task.tgt_dict, args.decoder_embed_dim
            )

        mt_encoder = cls.build_text_encoder(args, task.src_dict, encoder_embed_tokens)

        # speech_encoder = cls.build_speech_encoder(args)
        # text_encoder = cls.build_text_encoder(args, task.src_dict, encoder_embed_tokens)
        # text_decoder = cls.build_decoder(args, task.tgt_dict, decoder_embed_tokens)
        #
        # # freeze pretrained model
        # for param in speech_encoder.parameters():
        #     param.requires_grad = False
        # for param in text_decoder.parameters():
        #     param.requires_grad = False

        return cls(st_encoder, st_decoder, mt_encoder)

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def forward(self, src_tokens, src_lengths, prev_output_tokens, modal="speech", return_all_hiddens=False,
                features_only=False):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        # encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths, modal=modal)
        # TODO 将st_encoder和mt_encoder写入一个encoder，否则推理时无法调用mt_encoder
        encoder = self.st_encoder if modal == "speech" else self.mt_encoder
        encoder_out = encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out = self.st_decoder(
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            return_all_hiddens=return_all_hiddens,
            features_only=features_only,
        )
        return decoder_out


@register_model_architecture("s2t_jmt_transformer", "s2t_jmt_transformer")
def base_s2t_jmt_architecture(args):
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    base_architecture(args)


@register_model_architecture("s2t_jmt_transformer", "s2t_jmt_transformer_s")
def s2t_jmt_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_s2t_jmt_architecture(args)
