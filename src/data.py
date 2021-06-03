import os
import numpy as np
import pandas as pd
import h5py
import random
import torch
import torch.nn.functional as F

from fairseq.data import FairseqDataset, data_utils
from torch.utils.data.dataloader import default_collate

from PIL import Image


def split_file(split):
    return os.path.join('splits', f'karpathy_{split}_images.txt')


def read_split_image_ids_and_paths(split):
    split_df = pd.read_csv(split_file(split), sep=' ', header=None)
    return split_df.iloc[:, 1].to_numpy(), split_df.iloc[:, 0].to_numpy()


def read_split_image_ids(split):
    return read_split_image_ids_and_paths(split)[0]


def read_image_ids(file, source_only=False):
    with open(file, 'r') as f:
        image_ids = [int(line) for line in f]
    f.close()
    if source_only:
        image_ids = list(set(image_ids))
    return image_ids


def read_image_metadata(file):
    df = pd.read_csv(file)
    md = {}

    for img_id, img_h, img_w, num_boxes in zip(df['image_id'], df['image_h'], df['image_w'], df['num_boxes']):
        md[img_id] = {
            'image_h': np.float32(img_h),
            'image_w': np.float32(img_w),
            'num_boxes': num_boxes
        }

    return md


class SGDataset(torch.utils.data.Dataset):
    def __init__(self, sg_file, image_ids):
        self.sg_file = sg_file
        self.image_ids = image_ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        return self.read_data(self.image_ids[idx])

    def size(self):
        pass

    def read_data(self, image_id):
        h5py_dataset = h5py.File(self.sg_file, 'r', libver='latest')
        rel = h5py_dataset[str(image_id)+'-obj-rel'][()]
        obj_attr = h5py_dataset[str(image_id)+'-obj-attr'][()]
        obj = obj_attr[:, 1:4]
        attr = obj_attr[:, 4:]
        return torch.as_tensor(rel.astype(np.int64)), \
            torch.as_tensor(obj.astype(np.int64)), \
            torch.as_tensor(attr.astype(np.int64))

    def collater(self, samples):
        rel_lens, obj_lens, attr_lens = [], [], []
        for rel, obj, attr in samples:
            assert rel.shape[1] == 3 and obj.shape[1] == 3 and attr.shape[1] == 3
            rel_lens.append(rel.shape[0])
            obj_lens.append(obj.shape[0])
            attr_lens.append(attr.shape[0])
        max_rel_len, max_obj_len, max_attr_len, = max(rel_lens), max(obj_lens), max(attr_lens)

        padded_rels, padded_objs, padded_attrs = [], [], []
        for (rel, obj, attr), rel_len, obj_len, attr_len in zip(samples, rel_lens, obj_lens, attr_lens):
            assert len(rel.shape) == 2 and len(obj.shape) == 2 and len(attr.shape) == 2
            padded_rel = F.pad(rel, pad=[0, 0, 0, max_rel_len - rel_len], mode='constant', value=0.0)
            padded_obj = F.pad(obj, pad=[0, 0, 0, max_obj_len - obj_len], mode='constant', value=0.0)
            padded_attr = F.pad(attr, pad=[0, 0, 0, max_attr_len - attr_len], mode='constant', value=0.0)

            padded_rels.append(padded_rel)
            padded_objs.append(padded_obj)
            padded_attrs.append(padded_attr)

        return default_collate(padded_rels), default_collate(padded_objs), default_collate(padded_attrs)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, image_paths, transform=lambda x: x):
        self.image_ids = image_ids
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with Image.open(image_path).convert('RGB') as img:
            return self.transform(img), self.image_ids[idx]


class FeaturesDataset(FairseqDataset):
    def __init__(self, features_file, image_ids, num_objects, no_id=True):
        self.features_file = features_file
        self.image_ids = image_ids
        self.num_objects = num_objects
        self.no_id = no_id

    def __getitem__(self, index):
        return self.read_data(self.image_ids[index])

    def __len__(self):
        return len(self.image_ids)

    def num_tokens(self, index):
        return self.num_objects[index]

    def size(self, index):
        return self.num_objects[index]

    @property
    def sizes(self):
        return self.num_objects

    def read_data(self, image_id):
        raise NotImplementedError

    def collater(self, samples):
        num_objects = [features.shape[0] for features, _ in samples]
        max_objects = max(num_objects)

        feature_samples_padded = []
        location_samples_padded = []

        for (features, locations), n in zip(samples, num_objects):
            features_padded = F.pad(features, pad=[0, 0, 0, max_objects-n], mode='constant', value=0.0)
            locations_padded = F.pad(locations, pad=[0, 0, 0, max_objects-n], mode='constant', value=0.0)
            feature_samples_padded.append(features_padded)
            location_samples_padded.append(locations_padded)

        return default_collate(feature_samples_padded), default_collate(location_samples_padded)


class GridFeaturesDataset(FeaturesDataset):
    def __init__(self, features_file, image_ids, grid_shape=(8, 8), no_id=True):
        super().__init__(features_file=features_file,
                         image_ids=image_ids,
                         num_objects=np.ones(len(image_ids), dtype=np.int) * np.prod(grid_shape), no_id=no_id)

        self.grid_shape = grid_shape
        self.locations = self.tile_locations(grid_shape)

        # self.h5py_dataset = h5py.File(features_file, 'r')

    def read_data(self, image_id):

        # features_file = os.path.join(self.features_dir, f'{image_id}.npy')
        # features = np.load(features_file)
        h5py_dataset = h5py.File(self.features_file, 'r', libver='latest')

        features = h5py_dataset[str(image_id)][()]
        if self.no_id:
            return torch.as_tensor(features), self.locations
        else:
            return torch.as_tensor(features), self.locations, image_id

    @staticmethod
    def tile_locations(grid_shape):
        num_tiles = np.prod(grid_shape)
        rel_tile_w = 1. / grid_shape[1]
        rel_tile_h = 1. / grid_shape[0]
        rel_tile_area = 1. / num_tiles

        rel_tile_locations = np.zeros(shape=(grid_shape[0], grid_shape[1], 5), dtype=np.float32)

        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                rel_tile_locations[i, j] = np.array([
                    j * rel_tile_w,
                    i * rel_tile_h,
                    (j+1) * rel_tile_w,
                    (i+1) * rel_tile_h,
                    rel_tile_area
                ], dtype=np.float32)

        return torch.as_tensor(rel_tile_locations).view(num_tiles, 5)


class ObjectFeaturesDataset(FeaturesDataset):
    def __init__(self, features_file, image_ids, image_metadata, no_id=True):
        super().__init__(features_file=features_file,
                         image_ids=image_ids,
                         num_objects=np.array([image_metadata[image_id]['num_boxes'] for image_id in image_ids]),
                         no_id=no_id)

        self.image_metadata = image_metadata

    def read_data(self, image_id):
        # features_file = os.path.join(self.features_dir, f'{image_id}.npy')
        # features = np.load(features_file)
        #
        # boxes_file = os.path.join(self.features_dir, f'{image_id}-boxes.npy')
        # boxes = np.load(boxes_file)
        h5py_dataset = h5py.File(self.features_file, 'r', libver='latest')

        features = h5py_dataset[str(image_id)][()]
        boxes = h5py_dataset[str(image_id)+'-boxes'][()]

        # Normalize box coordinates
        boxes[:, [0, 2]] /= self.image_metadata[image_id]['image_w']
        boxes[:, [1, 3]] /= self.image_metadata[image_id]['image_h']

        # Normalized box areas
        areas = (boxes[:, 2] - boxes[:, 0]) * \
                (boxes[:, 3] - boxes[:, 1])

        if self.no_id:
            return torch.as_tensor(features), torch.as_tensor(np.c_[boxes, areas])
        else:
            return torch.as_tensor(features), torch.as_tensor(np.c_[boxes, areas]), image_id


class ImageCaptionDataset(FairseqDataset):
    def __init__(self, img_ds, cap_ds, sg_ds, cap_dict,
                 image_ids=None, shuffle=False, max_paraphrase_length=19):
        self.img_ds = img_ds
        self.cap_ds = cap_ds
        self.sg_ds = sg_ds
        self.cap_dict = cap_dict
        self.shuffle = shuffle
        self.image_ids = image_ids
        if image_ids is not None:
            image_id_to_group_indices = {}
            for idx, image_id in enumerate(image_ids):
                if image_id_to_group_indices.get(image_id) is not None:
                    image_id_to_group_indices[image_id].append(idx)
                else:
                    image_id_to_group_indices[image_id] = [idx]
            self.image_id_to_group_indices = image_id_to_group_indices
        else:
            self.image_id_to_group_indices = None
        self.max_paraphrase_length = max_paraphrase_length

    def __getitem__(self, index):
        max_paraphrase_len = getattr(self, 'max_paraphrase_length', 19)

        def get_paraphrase():
            if self.image_id_to_group_indices is not None and self.cap_ds is not None:
                group_indices = self.image_id_to_group_indices.get(self.image_ids[index])
                # print('| select image_ids: ', index, self.image_ids[index])
                # print('group_indices: ', group_indices)
                random.shuffle(group_indices)
                # paraphrase_index = random.choice(group_indices)
                paraphrase_captions = []
                paraphrase_lengths = 0
                for paraphrase_index in group_indices[:5]:
                    paraphrase_caption = self.cap_ds[paraphrase_index]
                    paraphrase_lengths += (self.cap_ds.sizes[paraphrase_index] + 1)
                    paraphrase_captions.append(
                        F.pad(paraphrase_caption[:max_paraphrase_len],
                              pad=[0, 1], mode='constant',
                              value=self.cap_dict.eos()))
                paraphrase_captions = torch.cat(paraphrase_captions, dim=0)
                # print('| paraphrase_captions: ', paraphrase_captions)
                dat = {
                    'paraphrase_index': group_indices[:5],
                    'paraphrase_caption': paraphrase_captions,
                    'paraphrase_length': paraphrase_lengths,
                }
            else:
                dat = {}
            return dat

        def get_cap():
            if self.cap_ds is not None:
                target = self.cap_ds[index]
                dat = {
                    'target': target,
                    'caption': target
                }
            else:
                dat = {}
            return dat

        def get_img():
            if self.img_ds.no_id:
                object_feature, object_location = self.img_ds[index]
                dat = {
                    'id': index,
                    'object_feature': object_feature,
                    'object_location': object_location,
                }
            else:
                object_feature, object_location, image_id = self.img_ds[index]
                dat = {
                    'id': index,
                    'image_id': image_id,
                    'object_feature': object_feature,
                    'object_location': object_location,
                }
            return dat

        def get_sg():
            if self.sg_ds is not None:
                rel, obj, attr = self.sg_ds[index]
                dat = {
                    "relation": rel,
                    "object": obj,
                    "attribute": attr
                }
            else:
                dat = {}
            return dat

        data = dict()
        cap = get_cap()
        img = get_img()
        sg = get_sg()
        para = get_paraphrase()
        data.update(cap)
        data.update(img)
        data.update(sg)
        data.update(para)
        return data

    def __len__(self):
        if self.cap_ds is not None:
            return len(self.cap_ds)
        else:
            return len(self.img_ds)

    def num_tokens(self, index):
        if self.cap_ds is not None:
            return self.size(index)[1]
        else:
            return self.img_ds.sizes[index]  # self.size(index)

    def size(self, index):
        # number of image feature vectors, number of tokens in caption
        if self.cap_ds is not None:
            return self.img_ds.sizes[index], self.cap_ds.sizes[index]
        else:
            return self.img_ds.sizes[index]  # self.img_ds.sizes[index]

    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))

        # Inspired by LanguagePairDataset.ordered_indices
        if self.cap_ds is not None:
            indices = indices[np.argsort(self.cap_ds.sizes[indices], kind='mergesort')]

        return indices[np.argsort(self.img_ds.sizes[indices], kind='mergesort')]

    def collater(self, samples):
        indices = []
        image_ids = []

        object_feature_samples = []
        object_location_samples = []
        object_lengths = []

        relation_samples = []
        object_samples = []
        attribute_samples = []
        relation_lengths = []

        target_samples = []
        target_ntokens = 0

        caption_samples, caption_lengths = [], []
        paraphrase_indices, paraphrase_samples, paraphrase_lengths = [], [], []

        for sample in samples:
            index = sample['id']
            indices.append(index)

            if sample.get('image_id') is not None:
                image_ids.append(sample['image_id'])

            object_feature_samples.append(sample['object_feature'])
            object_location_samples.append(sample['object_location'])
            object_lengths.append(self.img_ds.sizes[index])

            if sample.get('target') is not None:
                target_samples.append(sample['target'])
                target_ntokens += self.cap_ds.sizes[index]

            if sample.get('relation') is not None and \
                    sample.get('object') is not None and \
                    sample.get('attribute') is not None:
                relation_samples.append(sample['relation'])
                relation_length = sample['relation'].sum(-1).gt(0.0).long().sum().item()
                relation_lengths.append(relation_length)
                object_samples.append(sample['object'])
                attribute_samples.append(sample['attribute'])

            if sample.get('caption') is not None:
                caption_samples.append(sample['caption'])
                caption_lengths.append(self.cap_ds.sizes[index])

            if sample.get('paraphrase_index') is not None and sample.get('paraphrase_caption') is not None\
                    and sample.get('paraphrase_length') is not None:
                paraphrase_index = sample.get('paraphrase_index')
                paraphrase_indices.append(paraphrase_index)
                paraphrase_samples.append(sample.get('paraphrase_caption'))
                paraphrase_lengths.append(sample.get('paraphrase_length'))

        num_sentences = len(samples)

        # FIXME: workaround for edge case in parallel processing
        # (framework passes empty samples list
        # to collater under certain conditions)
        if num_sentences == 0:
            return None

        indices = torch.tensor(indices, dtype=torch.long)
        image_ids = torch.tensor(image_ids, dtype=torch.long)

        object_feature_batch, object_location_batch = \
            self.img_ds.collater(list(zip(object_feature_samples, object_location_samples)))

        if len(relation_samples) > 0 and len(object_samples) > 0 \
                and len(attribute_samples) > 0 and self.sg_ds is not None:
            relation_batch, object_batch, attribute_batch = \
                self.sg_ds.collater(list(zip(relation_samples, object_samples, attribute_samples)))
        else:
            relation_batch, object_batch, attribute_batch = None, None, None

        if self.cap_ds is not None:
            caption_tokens = data_utils.collate_tokens(
                caption_samples, pad_idx=self.cap_dict.pad(), eos_idx=self.cap_dict.eos(), move_eos_to_beginning=False)
        else:
            caption_tokens = None

        caption_lengths = torch.tensor(caption_lengths, dtype=torch.long)

        if self.cap_ds is not None:
            target_batch = data_utils.collate_tokens(target_samples, pad_idx=self.cap_dict.pad(), eos_idx=self.cap_dict.eos(), move_eos_to_beginning=False)
            rotate_batch = data_utils.collate_tokens(target_samples, pad_idx=self.cap_dict.pad(), eos_idx=self.cap_dict.eos(), move_eos_to_beginning=True)
        else:
            target_batch = None
            rotate_batch = None

        object_lengths = torch.tensor(object_lengths, dtype=torch.long)
        relation_lengths = torch.tensor(relation_lengths, dtype=torch.long)

        if len(paraphrase_samples) > 0 and len(paraphrase_lengths) > 0:
            paraphrase_samples = data_utils.collate_tokens(
                paraphrase_samples, pad_idx=self.cap_dict.pad(), eos_idx=self.cap_dict.eos(), move_eos_to_beginning=False)
            paraphrase_lengths = torch.tensor(paraphrase_lengths, dtype=torch.long)
            paraphrase_indices = torch.tensor(paraphrase_indices)
        else:
            paraphrase_samples, paraphrase_lengths, paraphrase_indices = None, None, None

        return {
            'id': image_ids,
            'net_input': {
                'object_features': object_feature_batch,
                'object_locations': object_location_batch,
                'object_lengths': object_lengths,
                'relations': relation_batch,
                'relation_lengths': relation_lengths,
                'objects': object_batch,
                'attributes': attribute_batch,
                'prev_output_tokens': rotate_batch,
                'caption_tokens': caption_tokens,
                'caption_lengths': caption_lengths,
                'paraphrase_tokens': paraphrase_samples,
                'paraphrase_lengths': paraphrase_lengths,
                'paraphrase_indices': paraphrase_indices
            },
            'target': target_batch,
            'ntokens': target_ntokens,
            'nsentences': num_sentences,
        }
