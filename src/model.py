# from .modules import *

import math
from typing import Any, Dict, List, NamedTuple, Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from torch import Tensor


from torchvision import models
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.inception import model_urls

from fairseq.models import FairseqEncoder, BaseFairseqModel
from fairseq.models import register_model, register_model_architecture, transformer
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from .mask_multihead_attention import MaskMultiheadAttention


def create_padding_mask(src_tokens, src_lengths):
    padding_mask = torch.zeros(src_tokens.shape[:2],
                               dtype=torch.bool,
                               device=src_tokens.device)

    for i, src_length in enumerate(src_lengths):
        padding_mask[i, src_length:] = 1

    return padding_mask


@register_model('gen2m2')
class gen2m2(BaseFairseqModel):
    @staticmethod
    def add_args(parser):
        transformer.TransformerModel.add_args(parser)
        parser.add_argument('--feature-dim', type=int, default=2048,
                            help='visual features dimension')
        parser.add_argument('--feature-spatial-encoding', default=False, action='store_true',
                            help='use feature spatial encoding')
        parser.add_argument('--super-obj-num', type=int, default=1,
                            help='number of super object nodes')
        parser.add_argument('--object-layernorm-embeddings', default=False, action='store_true',
                            help='layernorm the embedding of object')
        parser.add_argument('--n-object', type=int, default=472,
                            help='number of object label')
        parser.add_argument('--n-relation', type=int, default=472,
                            help='number of relation label')


    @classmethod
    def build_model(cls, args, task):
        transformer.base_architecture(args)

        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = transformer.DEFAULT_MAX_TARGET_POSITIONS

        captions_dict = task.target_dictionary
        if args.share_all_embeddings:
            assert args.encoder_embed_dim == args.decoder_embed_dim
            encoder_embedding = Embedding(len(captions_dict), args.encoder_embed_dim, captions_dict.pad())
            decoder_embedding = encoder_embedding 
        else:
            encoder_embedding = Embedding(len(captions_dict), args.encoder_embed_dim, captions_dict.pad())
            decoder_embedding = Embedding(len(captions_dict), args.decoder_embed_dim, captions_dict.pad())

        if args.n_relation == args.n_object:
            embed_object_labels = embed_relation_labels = Embedding(args.n_relation, args.encoder_embed_dim)
        else:
            embed_object_labels = Embedding(args.n_relation, args.encoder_embed_dim)
            embed_relation_labels = Embedding(args.n_relation, args.encoder_embed_dim)

        encoder = GenEncoder(args, captions_dict, encoder_embedding, embed_object_labels, embed_relation_labels)
        decoder = transformer.TransformerDecoder(args, captions_dict, decoder_embedding)
        return gen2m2(encoder, decoder)

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, sentence_ae=False, *args, **kwargs):
        model_output = {}
        image_encoder_out, sentence_encoder_out, super_encoder_out = self.encoder(sentence_ae=sentence_ae, **kwargs)
        image_decoder_out = self.decoder(encoder_out=image_encoder_out, **kwargs)
        model_output['image_out'] = image_decoder_out[0]
        model_output['ic_attn_scores'] = image_encoder_out.attn_scores
        if sentence_ae:
            super_decoder_out = self.decoder(encoder_out=super_encoder_out, **kwargs)
            model_output['super_out'] = super_decoder_out[0]
            model_output['image_super_node_out'] = image_encoder_out.super_out
            model_output['super_node_out'] = super_encoder_out.super_out
            model_output['cr_attn_scores'] = super_encoder_out.attn_scores
        return model_output

    def forward_encoder(self, sentence_ae=False, **kwargs):
        encoder_outs = self.encoder(sentence_ae=sentence_ae, **kwargs)
        if not sentence_ae:
            return encoder_outs[0]
        else:
            return encoder_outs

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def max_decoder_positions(self):
        return self.decoder.max_positions()


EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Tensor),  # B x T
        ("encoder_embedding", Tensor),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("super_out", Tensor),
        ("attn_scores", Tensor)
    ],
)


class GenEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens, embed_object_labels, embed_relation_labels):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout = args.dropout

        self.embed_dim = embed_dim = args.encoder_embed_dim
        self.max_source_positions = args.max_source_positions

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.object_feature_fn = torch.nn.Linear(args.feature_dim, embed_dim, True)
        if args.feature_spatial_encoding:
            self.object_bbox_fn = Linear(5, embed_dim)
        else:
            self.object_bbox_fn = None

        if args.layernorm_embeddings:
            self.layernorm_embeddings = LayerNorm(embed_dim)
        else:
            self.layernorm_embeddings = None

        self.padding_idx = embed_tokens.padding_idx
        self.embed_tokens = embed_tokens
        self.embed_positions = (
            PositionalEmbedding(
                self.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.embed_object_labels = embed_object_labels
        self.embed_relation_labels = embed_relation_labels

        self.super_obj_num = args.super_obj_num
        self.super_object_x = torch.nn.Parameter(torch.randn(args.super_obj_num, embed_dim) * embed_dim ** -0.5)

        self.embed_groups = Embedding(
            num_embeddings=4, embedding_dim=embed_dim
        )

        self.layer_wise_attention = getattr(args, "layer_wise_attention", False)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                GenEncoderLayer(args) for i in range(args.encoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward_object_embedding(
            self, object_features, object_locations, object_labels, *args, **kwargs):
        object_feature_x = self.object_feature_fn(object_features)
        object_location_x = self.object_bbox_fn(object_locations) * self.embed_scale \
            if self.object_bbox_fn is not None else 0.0
        object_word_x = self.embed_object_labels(object_labels)
        x = object_embedding = object_feature_x + object_location_x + object_word_x
        if self.layernorm_embeddings is not None:
            x = self.layernorm_embeddings(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, object_embedding

    def forward_relation_embedding(self, relation_labels):
        x = relation_embedding = self.embed_scale * self.embed_relation_labels(relation_labels)
        if self.layernorm_embeddings is not None:
            x = self.layernorm_embeddings(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, relation_embedding

    def forward_super_node_embedding(self, bsz):
        super_object_x = self.super_object_x.unsqueeze(0).repeat(bsz, 1, 1)
        x = F.dropout(super_object_x, p=self.dropout, training=self.training)
        return x, super_object_x

    def forward_word_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embeddings is not None:
            x = self.layernorm_embeddings(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward_embedding(
        self,
        object_features,
        object_locations,
        object_labels,
        relation_labels,
    ):
        bsz = object_features.size(0)
        super_x, super_embedding = self.forward_super_node_embedding(bsz)
        relation_x, relation_embedding = self.forward_relation_embedding(relation_labels)
        object_x, object_embedding = self.forward_object_embedding(
            object_features, object_locations, object_labels)
        x = torch.cat([super_x, object_x, relation_x], dim=1)
        embedding_x = torch.cat([super_embedding, object_embedding, relation_embedding], dim=1)
        return x, embedding_x

    @torch.no_grad()
    def build_structure_mask(self, object_labels, object_lengths, relation_triplets, relation_lengths):
        assert object_labels.size(0) == relation_triplets.size(0)
        bsz, n_object, n_relation = object_labels.size(0), object_labels.size(1), relation_triplets.size(1)
        relation_labels = relation_triplets[:, :, 2]
        mask_device = object_labels.device
        object_padding_mask = create_padding_mask(object_labels, object_lengths)
        relation_padding_mask = create_padding_mask(relation_labels, relation_lengths)
        super_padding_mask = torch.zeros(self.super_obj_num, device=mask_device, dtype=torch.bool).\
            unsqueeze(0).repeat(bsz, 1)
        padding_mask = torch.cat([super_padding_mask, object_padding_mask, relation_padding_mask], dim=-1)

        relation_to_object_masks = 1 - torch.zeros(bsz, n_relation, n_object, device=mask_device, dtype=torch.long).\
            scatter(2, relation_triplets[:, :, :2], 1).mul((1 - relation_padding_mask.long()).unsqueeze(-1))
        object_masks = torch.ones(bsz, n_object, n_object, device=mask_device, dtype=torch.long).\
            mul(object_padding_mask.long().unsqueeze(1))
        relation_masks = (1 - torch.eye(n_relation, device=mask_device, dtype=torch.long)).unsqueeze(0).repeat(bsz, 1, 1)

        structure_masks = torch.cat(
            [
                torch.cat([object_masks, relation_to_object_masks.permute(0, 2, 1)], dim=-1),
                torch.cat([relation_to_object_masks, relation_masks], dim=-1)
            ], dim=1
        )
        structure_masks = F.pad(
            structure_masks, pad=[self.super_obj_num, 0, self.super_obj_num, 0, 0, 0], mode='constant', value=0).\
            add(padding_mask.unsqueeze(1)).clamp(0, 1)

        super_ranges = torch.arange(
            self.super_obj_num, dtype=torch.long, device=mask_device)
        object_ranges = torch.arange(
            self.super_obj_num, self.super_obj_num+n_object, dtype=torch.long, device=mask_device)
        relation_ranges = torch.arange(
            self.super_obj_num+n_object, self.super_obj_num+n_object+n_relation, dtype=torch.long, device=mask_device)
        x_ranges = [super_ranges, object_ranges, relation_ranges]

        return structure_masks, padding_mask, x_ranges

    def forward(
        self,
        object_features,
        object_locations,
        object_lengths,
        objects,
        relations,
        relation_lengths,
        caption_tokens=None,
        caption_lengths=None,
        sentence_ae=False,
        return_all_hiddens: bool = False,
        **kwargs,
    ):
        if self.layer_wise_attention:
            return_all_hiddens = True

        bsz, dim_object, dim_relation, embed_device = objects.size(0), objects.size(1), relations.size(1), objects.device
        relation_triplets = relations
        object_labels, relation_labels = objects[:, :, 0], relations[:, :, 2]
        x, embedding_x = self.forward_embedding(
            object_features=object_features, object_locations=object_locations,
            object_labels=object_labels, relation_labels=relation_labels
        )
        group_embedding = self.embed_scale * self.embed_groups(
            torch.cat([torch.zeros(self.super_obj_num, device=embed_device, dtype=torch.long),
                       torch.ones(dim_object, device=embed_device, dtype=torch.long),
                       2 * torch.ones(dim_relation, device=embed_device, dtype=torch.long)], dim=0))
        x += group_embedding.unsqueeze(0)

        structure_mask, padding_mask, x_ranges = self.build_structure_mask(
            object_labels=object_labels, object_lengths=object_lengths,
            relation_triplets=relation_triplets, relation_lengths=relation_lengths
        )

        x = x.transpose(0, 1)

        # compute padding mask: B x T
        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        ic_attn_scores = []
        for i, layer in enumerate(self.layers):
            x, ic_attn_score = layer(x, padding_mask, structure_mask=structure_mask)
            ic_attn_scores.append(ic_attn_score)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)
        ic_attn_scores = torch.stack(ic_attn_scores, dim=1)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        image_encoder_out = EncoderOut(
            encoder_out=x,
            encoder_padding_mask=padding_mask,
            encoder_embedding=embedding_x,
            encoder_states=encoder_states,
            super_out=F.normalize(x[:self.super_obj_num], p=2, dim=-1),
            attn_scores=ic_attn_scores,
        )

        if caption_tokens is not None and caption_lengths is not None and sentence_ae:
            sentence_x, sentence_embedding = self.forward_word_embedding(caption_tokens)
            super_x, super_embedding = self.forward_super_node_embedding(bsz)
            x = embedding_x = torch.cat([super_x, sentence_x], dim=1)
            dim_sentence = sentence_x.size(1)
            group_embedding = self.embed_scale * self.embed_groups(
                torch.cat([torch.zeros(self.super_obj_num, device=embed_device, dtype=torch.long),
                           3 * torch.ones(dim_sentence, device=embed_device, dtype=torch.long)], dim=0))
            x += group_embedding.unsqueeze(0)

            sentence_padding_mask = create_padding_mask(caption_tokens, caption_lengths)
            super_padding_mask = torch.zeros(bsz, self.super_obj_num).type_as(sentence_padding_mask)
            padding_mask = torch.cat([super_padding_mask, sentence_padding_mask], dim=-1)
            x = x.transpose(0, 1)

            # compute padding mask: B x T
            encoder_states = [] if return_all_hiddens else None
            super_encoder_states = [] if return_all_hiddens else None

            cr_attn_scores = []
            # encoder layers
            for i, layer in enumerate(self.layers):
                x, cr_attn_score = layer(x, padding_mask)
                cr_attn_scores.append(cr_attn_score)
                if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(x)
                    super_encoder_states.append(x[:self.super_obj_num])
            cr_attn_scores = torch.stack(cr_attn_scores, dim=1)

            if self.layer_norm is not None:
                x = self.layer_norm(x)
                if return_all_hiddens:
                    encoder_states[-1] = x

            sentence_encoder_out = EncoderOut(
                encoder_out=x,
                encoder_padding_mask=padding_mask,
                encoder_embedding=embedding_x,
                encoder_states=encoder_states,
                super_out=F.normalize(x[:self.super_obj_num], p=2, dim=-1),
                attn_scores=cr_attn_scores,
            )
            super_sentence_encoder_out = EncoderOut(
                encoder_out=x[:self.super_obj_num],
                encoder_padding_mask=super_padding_mask,
                encoder_embedding=super_x,
                encoder_states=super_encoder_states,
                super_out=F.normalize(x[:self.super_obj_num], p=2, dim=-1),
                attn_scores=cr_attn_scores,
            )
            return image_encoder_out, sentence_encoder_out, super_sentence_encoder_out
        else:
            return image_encoder_out, None, None

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out.encoder_out is not None:
            encoder_out = encoder_out._replace(
                encoder_out=encoder_out.encoder_out.index_select(1, new_order)
            )
        if encoder_out.encoder_padding_mask is not None:
            encoder_out = encoder_out._replace(
                encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(0, new_order)
            )
        if encoder_out.encoder_embedding is not None:
            encoder_out = encoder_out._replace(
                encoder_embedding=encoder_out.encoder_embedding.index_select(0, new_order)
            )
        if encoder_out.encoder_states is not None:
            for idx, state in enumerate(encoder_out.encoder_states):
                encoder_out.encoder_states[idx] = state.index_select(1, new_order)
        return encoder_out


    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(
                    utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
                )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class GenEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)
        # self.embed_dim = args.encoder_embed_dim
        self.self_attn = MaskMultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout, self_attention=True
        )

    def forward(self, x, encoder_padding_mask, structure_mask=None, x_ranges=None,
                attn_mask: Optional[Tensor] = None):
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.bool(), -1e8)
        x, attn_score = self.self_attn(
            query=x, key=x, value=x, key_padding_mask=encoder_padding_mask,
            struct_mask=structure_mask, x_ranges=x_ranges)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x, attn_score


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture('gen2m2', 'gen2m2_small')
def gen2m2_small(args):
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.layernorm_embeddings = getattr(args, "layernorm_embeddings", True)
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.max_source_positions = getattr(args, "max_source_positions", 128)
    # args.max_image_positions = getattr(args, "max-image-positions", 100)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    # args.unified_encoder_layers = getattr(args, "unified_encoder_layers", 3)
    args.feature_dim = getattr(args, "feature_dim", 2048)
    args.feature_spatial_encoding = getattr(args, 'feature_spatial_encoding', True)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)


@register_model_architecture('gen2m2', 'gen2m2_base')
def gen2m2_base(args):
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.layernorm_embeddings = getattr(args, "layernorm_embeddings", True)
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.max_source_positions = getattr(args, "max_source_positions", 128)
    # args.max_image_positions = getattr(args, "max-image-positions", 100)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    # args.unified_encoder_layers = getattr(args, "unified_encoder_layers", 3)
    args.feature_dim = getattr(args, "feature_dim", 2048)
    args.feature_spatial_encoding = getattr(args, 'feature_spatial_encoding', True)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
