#!/usr/bin/env/python
import math



import torch
from torch import nn

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@register_criterion(
    "label_smoothed_cross_entropy_with_hidden_mapping")
class LabelSmoothedCrossEntropyCriterionWithHiddenMapping(LabelSmoothedCrossEntropyCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            hidden_mapping_loss_type="mse",
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.hidden_mapping_loss_type = hidden_mapping_loss_type

        # hidden mapping loss func
        self.hidden_mapping_loss = None
        if self.hidden_mapping_loss_type == "mse":
            self.hidden_mapping_loss = nn.MSELoss(reduction="none")

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)

        parser.add_argument(
            "--hidden-mapping-loss-type",
            default="mse",
            type=str,
            help="hidden mapping loss type: mse, kl, none..."
        )
        # parser.add_argument(
        #     "--hidden-mapping-task-flag",
        #     default=False,
        #     type=bool,
        #     help="Whether to perform the Hidden Mapping task."
        # )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_input = sample["net_input"]
        speech_net_input = {
            "src_tokens": net_input["src_tokens"]["src_audios"],
            "src_lengths": net_input["src_lengths"]["len_audios"],
            "prev_output_tokens": net_input["prev_output_tokens"],
            "modal": "speech",
        }
        speech_net_output = model(**speech_net_input)

        speech_loss, speech_nll_loss = self.compute_loss(model, speech_net_output, sample, reduce=reduce)

        text_loss, text_nll_loss, hidden_mapping_loss = torch.zeros(1).to(speech_loss.device), torch.zeros(1).to(
            speech_loss.device), torch.zeros(1).to(speech_loss.device)
        text_net_output = None
        if self.task.args.hidden_mapping_task_flag:
            text_net_input = {
                "src_tokens": net_input["src_tokens"]["src_texts"],
                "src_lengths": net_input["src_lengths"]["len_texts"],
                "prev_output_tokens": net_input["prev_output_tokens"],
                "modal": "text",
            }
            text_net_output = model(**text_net_input)

            text_loss, text_nll_loss = self.compute_loss(model, text_net_output, sample, reduce=reduce)

            target = sample["target"]  # [B, S]
            non_pad_mask = target.ne(self.padding_idx)  # [B, S], 1: not pad, 0: pad
            non_pad_mask = non_pad_mask.view(-1, 1)  # [B * S, 1]
            non_pad_idx = non_pad_mask.nonzero(as_tuple=True)[0]  # [ Select Size, 1]

            speech_decoder_hidden = speech_net_output[0]
            text_decoder_hidden = text_net_output[0]

            speech_decoder_hidden = \
                speech_decoder_hidden.reshape(-1, speech_decoder_hidden.size(-1))
            text_decoder_hidden = \
                text_decoder_hidden.reshape(-1, text_decoder_hidden.size(-1))

            speech_decoder_hidden = speech_decoder_hidden.index_select(dim=0, index=non_pad_idx)
            text_decoder_hidden = text_decoder_hidden.index_select(dim=0, index=non_pad_idx)

            hidden_mapping_loss = self.hidden_mapping_loss(
                speech_decoder_hidden, text_decoder_hidden)  # [ not pad size, H]

            hidden_mapping_loss = hidden_mapping_loss.sum(dim=-1).sum(dim=-1)
        loss = speech_loss + text_loss + hidden_mapping_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "hidden_mapping_loss": hidden_mapping_loss.data,
            "speech_loss": speech_loss.data,
            "speech_nll_loss": speech_nll_loss.data,
            "text_loss": text_loss.data,
            "text_nll_loss": text_nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, speech_net_output, sample)
            logging_output["speech_n_correct"] = utils.item(n_correct.data)
            logging_output["speech_total"] = utils.item(total.data)
            if self.task.args.hidden_mapping_task_flag:
                n_correct, total = self.compute_accuracy(model, text_net_output, sample)
                logging_output["text_n_correct"] = utils.item(n_correct.data)
                logging_output["text_total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs, ) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        hidden_mapping_loss_sum = sum(log.get("hidden_mapping_loss", 0) for log in logging_outputs)
        speech_loss_sum = sum(log.get("speech_loss", 0) for log in logging_outputs)
        speech_nll_loss_sum = sum(log.get("speech_nll_loss", 0) for log in logging_outputs)
        text_loss_sum = sum(log.get("text_loss", 0) for log in logging_outputs)
        text_nll_loss_sum = sum(log.get("text_nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "hidden_mapping_loss", hidden_mapping_loss_sum / sample_size / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "speech_loss", speech_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "speech_nll_loss", speech_nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "speech_ppl", lambda meters: utils.get_perplexity(meters["speech_nll_loss"].avg)
        )
        speech_total = utils.item(sum(log.get("speech_total", 0) for log in logging_outputs))
        if speech_total > 0:
            metrics.log_scalar("speech_total", speech_total)
            speech_n_correct = utils.item(
                sum(log.get("speech_n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("speech_n_correct", speech_n_correct)
            metrics.log_derived(
                "speech_accuracy",
                lambda meters: round(
                    meters["speech_n_correct"].sum * 100.0 / meters["speech_total"].sum, 3
                )
                if meters["speech_total"].sum > 0
                else float("nan"),
            )
        # if hidden_mapping_task_flag:
        metrics.log_scalar(
            "text_loss", text_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "text_nll_loss", text_nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "text_ppl", lambda meters: utils.get_perplexity(meters["text_nll_loss"].avg)
        )
        text_total = utils.item(sum(log.get("text_total", 0) for log in logging_outputs))
        if speech_total > 0:
            metrics.log_scalar("text_total", text_total)
            text_n_correct = utils.item(
                sum(log.get("text_n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("text_n_correct", text_n_correct)
            metrics.log_derived(
                "text_accuracy",
                lambda meters: round(
                    meters["text_n_correct"].sum * 100.0 / meters["text_total"].sum, 3
                )
                if meters["text_total"].sum > 0
                else float("nan"),
            )
