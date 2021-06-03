# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import (
    FairseqIncrementalDecoder,
    transformer
)


class SCSTGenerator(object):
    def __init__(
        self,
        tgt_dict,
        scb_beam_size=1,
        scr_beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.,
        unk_penalty=0.,
        retain_dropout=False,
        sampling=False,
        sampling_topk=-1,
        sampling_topp=-1.0,
        temperature=1.,
        diverse_beam_groups=-1,
        diverse_beam_strength=0.5,
        match_source_len=False,
        no_repeat_ngram_size=0,
    ):
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        # the max beam size is the dictionary size - 1, since we never select pad
        self.scb_beam_size = min(scb_beam_size, self.vocab_size - 1)
        self.scr_beam_size = min(scr_beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'
        assert sampling_topp < 0 or sampling, '--sampling-topp requires --sampling'
        assert temperature > 0, '--temperature must be greater than 0'
        self.scr_search = search.Sampling(tgt_dict, sampling_topk, sampling_topp)

    def generate(self, models, sample, **kwargs):
        scr_encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k not in ['prev_output_tokens', 'str_targets'] and isinstance(v, torch.Tensor)
        }
        scr_model = EnsembleModel(models)
        scr_model.train()
        # compute the encoder output for each beam
        scr_encoder_outs = scr_model.forward_encoder(scr_encoder_input)
        scr_tokens, ys_lprobs = self._scr_generate(scr_model, sample, scr_encoder_outs)
        return scr_tokens, ys_lprobs

    def _scr_generate(
        self,
        model,
        sample,
        encoder_outs=None,
        prefix_tokens=None,
        bos_token=None,
        **kwargs
    ):
        src_tokens = encoder_padding_mask = encoder_outs[0].encoder_padding_mask
        src_lengths = (1 - encoder_padding_mask.long()).sum(-1)
        # batch dimension goes first followed by source lengths
        bsz = encoder_padding_mask.size()[0]
        src_len = encoder_padding_mask.size()[1]
        beam_size = self.scr_beam_size

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                model.max_decoder_positions() - 1,
            )
        # compute the encoder output for each beam
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = model.reorder_encoder_out(encoder_outs, new_order)
        # initialize buffers
        tokens = src_tokens.new(bsz * beam_size, max_len + 2).long().fill_(self.pad)
        tokens[:, 0] = self.eos if bos_token is None else bos_token

        ys_lprobs = []
        hypo_tokens = []
        finishes = src_tokens.new(bsz * beam_size).bool().fill_(False)
        for step in range(max_len):
            y_lprobs, avg_attn_scores = model.forward_decoder(
                tokens[:, :step + 1].clone(), encoder_outs, temperature=self.temperature)
            lprobs = y_lprobs.clone().detach()
            # lprobs[:, self.pad] = -math.inf
            lprobs[:, self.unk] -= self.unk_penalty
            # handle max length constraint
            if step >= max_len:
                lprobs[:, :self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf
            if step < self.min_len:
                lprobs[:, self.eos] = -math.inf
            step_tokens = torch.distributions.Categorical(logits=lprobs.detach()).sample()
            hypo_tokens.append(step_tokens)
            tokens[:, step + 1] = step_tokens
            ys_lprobs.append(y_lprobs.gather(1, step_tokens.unsqueeze(1)).squeeze(-1))
            finishes += (step_tokens == self.eos)
            if finishes.sum().item() == bsz * beam_size:
                break

        ys_lprobs = torch.stack(ys_lprobs, dim=1)
        hypo_tokens = torch.stack(hypo_tokens, dim=1)
        masks = hypo_tokens.eq(self.eos).float().cumsum(-1).cumsum(-1).le(1.0).type_as(hypo_tokens)
        hypo_tokens = hypo_tokens.mul(masks).add((1 - masks).mul(self.pad))
        return hypo_tokens, ys_lprobs


class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.incremental_states = None
        if all(isinstance(m.decoder, FairseqIncrementalDecoder) for m in models):
            self.incremental_states = {m: {} for m in models}

    def has_encoder(self):
        return hasattr(self.models[0], 'encoder')

    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    def forward_encoder(self, encoder_input):
        if not self.has_encoder():
            return None
        return [model.forward_encoder(**encoder_input) for model in self.models]

    def forward_decoder(self, tokens, encoder_outs, temperature=1.):
        if len(self.models) == 1:
            return self._decode_one(
                tokens,
                self.models[0],
                encoder_outs[0] if self.has_encoder() else None,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )

        log_probs = []
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn = self._decode_one(
                tokens,
                model,
                encoder_out,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn

    def _decode_one(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1.,
    ):
        if self.incremental_states is not None:
            decoder_out = list(model.forward_decoder(
                tokens, encoder_out=encoder_out, incremental_state=self.incremental_states[model],
            ))
        else:
            decoder_out = list(model.forward_decoder(tokens, encoder_out=encoder_out))
        decoder_out[0] = decoder_out[0][:, -1:, :].contiguous() # .clone()
        if temperature != 1.:
            decoder_out[0] = decoder_out[0].div(temperature)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn.get('attn', None)
        if attn is not None:
            attn = attn[:, -1, :].contiguous().clone()
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :].contiguous() # .clone()
        return probs, attn

    def reorder_encoder_out(self, encoder_outs, new_order):
        if not self.has_encoder():
            return
        return [
            model.encoder.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]

    def reorder_incremental_state(self, new_order):
        if self.incremental_states is None:
            return
        for model in self.models:
            model.decoder.reorder_incremental_state(self.incremental_states[model], new_order)
