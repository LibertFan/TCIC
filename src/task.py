import os
from .data import *

from fairseq import search
from fairseq.data import Dictionary, data_utils
from fairseq.tasks import FairseqTask, register_task

# Import for registration of captioning model
# and architecture at fairseq registry.


@register_task('captioning')
class CaptioningTask(FairseqTask):
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

    @classmethod
    def setup_task(cls, args, **kwargs):
        captions_dict_file = os.path.join(args.captions_dir, f'dict.{args.captions_lang}.txt')
        captions_dict = Dictionary.load(captions_dict_file)

        return CaptioningTask(args, captions_dict)

    def __init__(self, args, captions_dict):
        super().__init__(args)
        self.captions_dict = captions_dict

    def load_dataset(self, split, **kwargs):
        print('| Directory of Features: {}'.format(self.args.features_dir))
        features_file = os.path.join(self.args.features_dir, f'{split}-{self.args.features}.h5py')

        if not self.args.source_only:
            print('| Directory of Captions: {}'.format(self.args.captions_dir))
            captions_file = os.path.join(self.args.captions_dir, f'{split}-captions.{self.args.captions_lang}')
            captions_ds = data_utils.load_indexed_dataset(captions_file, self.captions_dict)
        else:
            captions_ds = None

        image_ids_file = os.path.join(self.args.captions_dir, f'{split}-ids.txt')
        image_ids = read_image_ids(image_ids_file, source_only=self.args.source_only)

        if self.args.features == 'grid':
            image_ds = GridFeaturesDataset(features_file, image_ids, grid_shape=(8, 8), no_id=not self.args.source_only)
        elif self.args.features == 'obj':
            image_metadata_file = os.path.join(self.args.features_dir, f'{split}-metadata.csv')
            image_metadata = read_image_metadata(image_metadata_file)
            image_ds = ObjectFeaturesDataset(features_file, image_ids, image_metadata, no_id=not self.args.source_only)
        else:
            raise ValueError(f'Invalid --features option: {self.args.features}')

        if self.args.sg_dir is not None and self.args.features == 'obj':
            print('| Directory of SceneGraph: {}'.format(self.args.sg_dir))
            sg_file = os.path.join(self.args.sg_dir, f'{split}-obj-rel-attr.h5py')
            scenegraph_ds = SGDataset(sg_file, image_ids)
        else:
            scenegraph_ds = None

        self.datasets[split] = ImageCaptionDataset(
            image_ds, captions_ds, scenegraph_ds, self.captions_dict, image_ids=image_ids,
            shuffle=True if not self.args.source_only else False)

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
                  gradient
                - logging outputs to display while training
        """
        model.train()
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return self.captions_dict

    def build_generator(self, args):

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
            len_penalty=getattr(args, 'lenpen', 1),
            unk_penalty=getattr(args, 'unkpen', 0),
            temperature=getattr(args, 'temperature', 1.),
            match_source_len=getattr(args, 'match_source_len', False),
            no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            search_strategy=search_strategy,
        )
