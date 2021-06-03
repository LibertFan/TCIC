import os
import numpy as np
import torch
from collections import OrderedDict

from .data_scst import *
from .scst_generator import SCSTGenerator

from fairseq.data import encoders
from fairseq import search
from fairseq import utils
from fairseq.data import Dictionary, data_utils
from fairseq.tasks import FairseqTask, register_task

import copy
import math
import warnings
import time

warnings.filterwarnings('ignore')

# Import for registration of captioning model
# and architecture at fairseq registry.


@register_task('scst_captioning')
class SCSTCaptioning(FairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--features', default='grid', choices=['grid', 'obj'],
                            help='image features')
        parser.add_argument('--features-dir', default='data-bin',
                            help='image features directory')
        parser.add_argument('--captions-dir', default='data-bin',
                            help='image captions directory')
        parser.add_argument('--sg-dir', default=None,
                            help='scene graphs directory')
        parser.add_argument('--captions-lang', default='en', choices=['en'],
                            help='caption language')
        parser.add_argument('--max-source-positions', default=64, type=int, metavar='N',
                            help='max number of objects in the source image')
        parser.add_argument('--max-target-positions', default=128, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--source-only', action='store_true',
                            help='load target or caption dataset')

        parser.add_argument('--cider-reward-weight', type=float, default=1,
                            help='The reward weight from cider')
        parser.add_argument('--bleu-reward-weight', type=float, default=0,
                            help='The reward weight from bleu4')
        parser.add_argument('--cached-tokens', type=str, default='coco_idx',
                            help='Cached tokens for Cider/Cider-D computation')
        parser.add_argument('--valid-cached-tokens', type=str, default='coco_test_idx',
                            help='Cached tokens for Cider/Cider-D validation computation')

        # fmt: off
        parser.add_argument('--scb-beam', default=5, type=int, metavar='N',
                            help='beam size for self-critic baseline')
        parser.add_argument('--scr-beam', default=5, type=int, metavar='N',
                            help='beam size of self-critic random')
        try:
            parser.add_argument('--max-len-a', default=0, type=float, metavar='N',
                                help=('generate sequences of maximum length ax + b, '
                                      'where x is the source length'))
        except:
            pass
        try:
            parser.add_argument('--max-len-b', default=200, type=int, metavar='N',
                                help=('generate sequences of maximum length ax + b, '
                                      'where x is the source length'))
        except:
            pass
        try:
            parser.add_argument('--min-len', default=1, type=float, metavar='N',
                                help='minimum generation length')
        except:
            pass
        try:
            parser.add_argument('--match-source-len', default=False, action='store_true',
                                help='generations should match the source length')
        except:
            pass
        try:
            parser.add_argument('--lenpen', default=1, type=float,
                                help='length penalty: <1.0 favors shorter, >1.0 favors longer sentences')
        except:
            pass
        try:
            parser.add_argument('--unkpen', default=0, type=float,
                                help='unknown word penalty: <0 produces more unks, >0 produces fewer')
        except:
            pass
        try:
            parser.add_argument('--replace-unk', nargs='?', const=True, default=None,
                                help='perform unknown replacement (optionally with alignment dictionary)')
        except:
            pass
        try:
            parser.add_argument('--prefix-size', default=0, type=int, metavar='PS',
                                help='initialize generation by target prefix of given length')
        except:
            pass
        try:
            parser.add_argument('--no-repeat-ngram-size', default=0, type=int, metavar='N',
                                help='ngram blocking such that this size ngram cannot be repeated in the generation')
        except:
            pass
        try:
            parser.add_argument('--sampling', action='store_true',
                                help='sample hypotheses instead of using beam search')
        except:
            pass
        try:
            parser.add_argument('--sampling-topk', default=-1, type=int, metavar='PS',
                                help='sample from top K likely next words instead of all words')
        except:
            pass
        try:
            parser.add_argument('--sampling-topp', default=-1.0, type=float, metavar='PS',
                                help='sample from the smallest set whose cumulative probability mass '
                                     'exceeds p for next words')
        except:
            pass
        try:
            parser.add_argument('--temperature', default=1., type=float, metavar='N',
                                help='temperature for generation')
        except:
            pass
        try:
            parser.add_argument('--remove-bpe', nargs='?', const='@@ ', default=None,
                               help='remove BPE tokens before scoring (can be set to sentencepiece)')
        except:
            pass

    @classmethod
    def setup_task(cls, args, **kwargs):
        captions_dict_file = os.path.join(args.captions_dir, f'dict.{args.captions_lang}.txt')
        captions_dict = Dictionary.load(captions_dict_file)

        return SCSTCaptioning(args, captions_dict)

    def __init__(self, args, captions_dict):
        super().__init__(args)
        # assert args.scr_beam > 1
        self.captions_dict = captions_dict
        torch.autograd.set_detect_anomaly(True)
        self.sc_generator, self.scorers, self.valid_scorer, self.sc_tokenizer, self.sc_decoder, \
            self.res_ptb_tokenizer, self.gts_ptb_tokenizer = self.setup_sc(args)
        self.generator = self.build_generator(args)

    def setup_sc(self, args):
        generator = SCSTGenerator(
            self.target_dictionary,
            scr_beam_size=getattr(args, 'scr_beam', 5),
            scb_beam_size=getattr(args, 'scb_beam', 5),
            max_len_a=getattr(args, 'max_len_a', 0),
            max_len_b=getattr(args, 'max_len_b', 200),
            min_len=getattr(args, 'min_len', 1),
            normalize_scores=(not getattr(args, 'unnormalized', False)),
            len_penalty=getattr(args, 'lenpen', 0.0),
            unk_penalty=getattr(args, 'unkpen', 0),
            sampling=getattr(args, 'sampling', False),
            sampling_topk=getattr(args, 'sampling_topk', -1),
            sampling_topp=getattr(args, 'sampling_topp', -1.0),
            temperature=getattr(args, 'temperature', 1.),
            diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
            diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
            match_source_len=getattr(args, 'match_source_len', False),
            no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
        )

        import sys
        sys.path.append("cider")
        from pyciderevalcap.ciderD.ciderD import CiderD
        from pyciderevalcap.cider.cider import Cider
        from pyciderevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        sys.path.append("coco-caption")
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge

        print('| Cached Tokens: {}'.format(args.cached_tokens))
        CiderD_scorer = CiderD(df=args.cached_tokens)
        Cider_scorer = Cider(df=args.cached_tokens)
        Bleu_scorer = Bleu(4)
        meteor_scorer = Meteor()
        rouge_scorer = Rouge()

        print('| args.valid_cached_tokens: ', args)
        valid_Cider_scorer = Cider(df=args.valid_cached_tokens)

        sc_tokenizer = encoders.build_tokenizer(args)
        sc_decoder = encoders.build_bpe(args)

        # Tokenizer needed for computing CIDEr scores
        res_ptb_tokenizer = PTBTokenizer('res')
        gts_ptb_tokenizer = PTBTokenizer('gts')
        print('| setup_sc | scr_beam: ', args.scr_beam)

        return generator, {"CiderD": CiderD_scorer, 'Cider': Cider_scorer, 'BLEU4': Bleu_scorer,
                           "METEOR": meteor_scorer, "ROUGE": rouge_scorer}, \
            valid_Cider_scorer, sc_tokenizer, sc_decoder, res_ptb_tokenizer, gts_ptb_tokenizer

    def load_dataset(self, split, **kwargs):

        image_ids_file = os.path.join(self.args.captions_dir, f'{split}-ids.txt')
        image_ids = read_image_ids(image_ids_file, source_only=True)

        if not self.args.source_only:
            print('| Directory of Captions: {}'.format(self.args.captions_dir))
            captions_file = os.path.join(self.args.captions_dir, f'{split}-ptb.json')
            captions_ds = SCCaptionDataset(captions_file, image_ids)
        else:
            captions_ds = None

        print('| Directory of Features: {}'.format(self.args.features_dir))
        features_file = os.path.join(self.args.features_dir, f'{split}-{self.args.features}.h5py')

        if self.args.features == 'grid':
            image_ds = GridFeaturesDataset(features_file, image_ids, grid_shape=(8, 8))
        elif self.args.features == 'obj':
            image_metadata_file = os.path.join(self.args.features_dir, f'{split}-metadata.csv')
            image_metadata = read_image_metadata(image_metadata_file)
            image_ds = ObjectFeaturesDataset(features_file, image_ids, image_metadata)
        else:
            raise ValueError(f'Invalid --features option: {self.args.features}')

        if self.args.sg_dir is not None and self.args.features == 'obj':
            print('| Directory of SceneGraph: {}'.format(self.args.sg_dir))
            sg_file = os.path.join(self.args.sg_dir, f'{split}-obj-rel-attr.h5py')
            scenegraph_ds = SGDataset(sg_file, image_ids)
        else:
            scenegraph_ds = None

        self.datasets[split] = ImageCaptionDataset(image_ds, captions_ds, scenegraph_ds, self.captions_dict,
                                                   shuffle=True if not self.args.source_only else False)

    def decode(self, x):
        """Decode model output.
        """
        x = self.target_dictionary.string(x)
        x = self.sc_decoder.decode(x)
        return self.sc_tokenizer.decode(x)

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient2
                - logging outputs to display while training
        """
        args = self.args
        tgt_dict = self.captions_dict

        model.train()
        scr_hypos, ys_lprobs = self.sc_generator.generate([model], sample)

        seq_per_img = args.scr_beam
        batch_size = len(sample['id'].tolist())
        hypo_size = batch_size * seq_per_img

        res = OrderedDict()
        gts = OrderedDict()
        for i, sample_id in enumerate(sample['id'].tolist()):
            target_str = sample['str_targets'][i]

            for j, hypo in enumerate(scr_hypos[i*seq_per_img: (i+1)*seq_per_img]):
                hypo_len = hypo.ne(self.target_dictionary.pad()).sum().item()
                hypo = hypo[:hypo_len]
                scr_hypo_str = self.decode(hypo.int().cpu())
                res[seq_per_img*i+j] = scr_hypo_str
                gts[seq_per_img*i+j] = target_str

        for i in range(seq_per_img):
            print('| res[{}]: '.format(i), res.get(i))
        # print('| res2: ', res.get(1))
        res_cider = [{'image_id': i, 'caption': res[i]} for i in range(len(res))]
        res_cider = self.res_ptb_tokenizer.tokenize(res_cider)

        if self.args.cider_reward_weight > 0:
            cider_score, cider_scores = self.scorers['Cider'].compute_score(gts, res_cider)
            cider_scores = np.array(cider_scores)
        else:
            cider_scores = 0
        if self.args.bleu_reward_weight > 0:
            res_bleu = {i: res_cider[i]['caption'] for i in range(len(res_cider))}
            _, bleu_scores = self.scorers['BLEU4'].compute_score(gts, res_bleu)
            _, meteor_scores = self.scorers['METEOR'].compute_score(gts, res_bleu)
            _, rouge_scores = self.scorers['ROUGE'].compute_score(gts, res_bleu)
            bleu_scores = np.array(bleu_scores[3])
            meteor_scores = np.array(meteor_scores)
            rouge_scores = np.array(rouge_scores)
            print('| BLEU: {} | METEOR: {} | ROUGE: {} |'.format(
                bleu_scores.mean(), meteor_scores.mean(), rouge_scores.mean()))
            bleu_scores += (meteor_scores + rouge_scores)
        else:
            bleu_scores = 0
        scores = scr_scores = self.args.cider_reward_weight * cider_scores + \
            self.args.bleu_reward_weight * bleu_scores
        scores = scores.reshape(batch_size, seq_per_img)
        baseline_scores = (np.sum(scores, axis=-1, keepdims=True) - scores) / float(seq_per_img - 1)
        scores = scores - baseline_scores
        scores = scores.reshape(hypo_size)

        rewards = np.repeat(scores[:, np.newaxis], scr_hypos.shape[1], 1)
        rewards = torch.as_tensor(rewards).type_as(ys_lprobs)
        # print('| rewards: ', rewards.size())
        loss, sample_size, logging_output = criterion(scr_hypos, ys_lprobs, rewards, sample, scr_scores)

        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):

        args = self.args
        tgt_dict = self.captions_dict

        model.eval()
        hypos = self.generator.generate([model], sample)

        batch_size = len(sample['id'].tolist())
        hypo_tokens = []
        nsentences = 0
        ntokens = 0

        res = OrderedDict()
        gts = OrderedDict()
        for i, sample_id in enumerate(sample['id'].tolist()):
            # Remove padding
            target_str = sample['str_targets'][i]
            gts[i] = target_str

            hypo = hypos[i][0]['tokens'].int().cpu()
            hypo_len = hypo.ne(self.target_dictionary.pad()).sum().item()
            hypo = hypo[:hypo_len]
            hypo_str = self.decode(hypo.int().cpu())
            res[i] = hypo_str
            nsentences += 1
            ntokens += hypo_len

        print('| res: ', res[0])
        res_cider = [{'image_id': i, 'caption': res[i]} for i in range(len(res))]
        res_cider = self.res_ptb_tokenizer.tokenize(res_cider)
        # gts_cider = {i: [{'caption': gt} for gt in gts[i]] for i in range(batch_size)}
        # gts_cider = self.gts_ptb_tokenizer.tokenize(gts_cider)
        cider, cider_scores = self.valid_scorer.compute_score(gts, res_cider)
        print("| valid cider: ", cider, np.array(cider_scores).mean())
        sample_size = nsentences
        logging_output = {
            'cider': np.sum(np.array(cider_scores)),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return cider, sample_size, logging_output

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return self.captions_dict

    def build_generator(self, args):
        print('| Task SCST build_generator')
        from .generator_base import BaseGenerator as SequenceGenerator
        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, 'sampling', False)
        sampling_topk = getattr(args, 'sampling_topk', -1)
        sampling_topp = getattr(args, 'sampling_topp', -1.0)
        diverse_beam_groups = getattr(args, 'diverse_beam_groups', -1)
        diverse_beam_strength = getattr(args, 'diverse_beam_strength', 0.5),
        match_source_len = getattr(args, 'match_source_len', False)
        diversity_rate = getattr(args, 'diversity_rate', -1)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError('Provided Search parameters are mutually exclusive.')
        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'
        assert sampling_topp < 0 or sampling, '--sampling-topp requires --sampling'

        if sampling:
            search_strategy = search.Sampling(self.target_dictionary, sampling_topk, sampling_topp)
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength)
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary, min_len_a=1, min_len_b=0, max_len_a=1, max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(self.target_dictionary, diversity_rate)
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            self.target_dictionary,
            beam_size=getattr(args, 'beam', 5),
            max_len_a=getattr(args, 'max_len_a', 0),
            max_len_b=getattr(args, 'max_len_b', 200),
            min_len=getattr(args, 'min_len', 1),
            normalize_scores=(not getattr(args, 'unnormalized', False)),
            len_penalty=getattr(args, 'lenpen', 0.0),
            unk_penalty=getattr(args, 'unkpen', 0),
            temperature=1.0,
            match_source_len=getattr(args, 'match_source_len', False),
            no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 3),
            search_strategy=search_strategy,
        )

