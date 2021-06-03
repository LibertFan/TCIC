import math
import torch
import numpy as np

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion, label_smoothed_cross_entropy


@register_criterion('base_scst')
class BaseSCST(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

    def forward(self, tokens, lprobs, rewards, sample, scr_scores, reduce=True, **kwargs):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        loss, sc_loss = self.compute_loss(tokens, lprobs, rewards)
        try:
            n_pos = rewards.ge(0.0).sum().div(rewards.size(1)).item()
        except:
            n_pos = rewards.ge(0.0).sum().floor_divide(rewards.size(1)).item()

        nsentences = tokens.size(0)
        ntokens = tokens.ne(self.padding_idx).sum().item()
        sample_size = nsentences if self.args.sentence_avg else ntokens

        logging_output = {
            'cider': np.sum(scr_scores) if reduce else scr_scores,
            'loss': utils.item(loss.data) if reduce else loss.data,
            'sc_loss': utils.item(sc_loss.data) if reduce else sc_loss.data,
            'n_positive': n_pos,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, tokens, lprobs, rewards, reduce=True):
        lprobs = lprobs.view(-1, 1)
        tokens = tokens.view(-1, 1)#.detach()
        rewards = rewards.view(-1, 1)#.detach()
        sc_loss = -lprobs.mul(rewards)
        pad_mask = tokens.eq(self.padding_idx)
        if pad_mask.any():
            sc_loss.masked_fill_(pad_mask, 0.)
        if reduce:
            sc_loss = sc_loss.sum()

        loss = sc_loss
        if self.eps > 0:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            if pad_mask.any():
                smooth_loss.masked_fill_(pad_mask, 0.)
            if reduce:
                smooth_loss = smooth_loss.sum()
            eps_i = self.eps / lprobs.size(-1)
            loss = (1. - self.eps) * sc_loss + eps_i * smooth_loss

        return loss, sc_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        cider_sum = sum(log.get('cider', 0) for log in logging_outputs)
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        sc_loss_sum = sum(log.get('sc_loss', 0) for log in logging_outputs)
        n_pos_sum = sum(log.get('n_positive', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        # print('| n_pos_sum: {} | nsentences: {}'.format(n_pos_sum, nsentences))
        agg_output = {
            'cider': cider_sum / nsentences,
            'loss': loss_sum / sample_size / math.log(2),
            'sc_loss': sc_loss_sum / ntokens / math.log(2) if ntokens > 0 else 0.,
            'n_positive_rate': n_pos_sum / nsentences,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
