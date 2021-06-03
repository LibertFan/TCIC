import math

from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('loss_tcic')
class LossTCIC(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.lambda_super = args.lambda_super
        self.lambda_align = args.lambda_align

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--lambda-super', default=0., type=float, metavar='D',
                            help='lambda for the loss of super generation loss')
        parser.add_argument('--lambda-align', default=0., type=float, metavar='D',
                            help='lambda for the l2 align')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        net_output = model(sentence_ae=True, **sample['net_input'])
        loss, loss_output = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'image_loss': utils.item(loss_output['image_loss'].data) \
                if reduce else loss_output['image_loss'].data,
            'image_nll_loss': utils.item(loss_output['image_nll_loss'].data) \
                if reduce else loss_output['image_nll_loss'].data,
            'super_loss': utils.item(loss_output['super_loss'].data) \
                if reduce else loss_output['super_loss'].data,
            'super_nll_loss': utils.item(loss_output['super_nll_loss'].data) \
                if reduce else loss_output['super_nll_loss'].data,
            'align_loss': utils.item(loss_output['align_loss'].data) \
                if reduce else loss_output['align_loss'].data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        loss_output = {}
        model_loss = 0.0
        if net_output.get('image_out') is not None:
            lprobs = net_output['image_out'].softmax(-1).log()
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = sample['target'].view(-1, 1)
            # print('| compute_loss | image_out: ', lprobs.size(), target.size())
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )
            loss_output['image_loss'] = loss
            loss_output['image_nll_loss'] = nll_loss
            model_loss += loss.sum()
        if net_output.get('super_out') is not None:
            lprobs = net_output['super_out'].softmax(-1).log()
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = sample['target'].view(-1, 1)
            # print('| compute_loss | super_out: ', lprobs.size(), target.size())
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )
            loss_output['super_loss'] = loss
            loss_output['super_nll_loss'] = nll_loss
            model_loss += self.lambda_super * loss
        if net_output.get('image_super_node_out') is not None and \
                net_output.get('super_node_out') is not None:
            align_loss = net_output.get('image_super_node_out').sub(net_output.get('super_node_out')).\
                pow(2).sum(-1).mean(0)
            align_loss = align_loss.sum()
            loss_output['align_loss'] = align_loss
            model_loss += self.lambda_align * align_loss

        return model_loss, loss_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size if sample_size > 0 else 0.,
            'image_loss': sum(log.get('image_loss', 0) for log in logging_outputs) / sample_size
                if sample_size > 0 else 0.,
            'image_nll_loss': sum(log.get('image_nll_loss', 0) for log in logging_outputs) / ntokens / math.log(
                2) if ntokens > 0 else 0.,
            'super_loss': sum(log.get('super_loss', 0) for log in logging_outputs) / sample_size
                if sample_size > 0 else 0.,
            'super_nll_loss': sum(log.get('super_nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2)
                if ntokens > 0 else 0.,
            'align_loss': sum(log.get('align_loss', 0) for log in logging_outputs) / sample_size
                if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
