import re
import os
import argparse

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def main():
    parser = get_parser()
    args = parser.parse_args()
    coco = COCO(args.reference_captions)
    coco_system_captions = coco.loadRes(args.system_captions)
    coco_eval = COCOEvalCap(coco, coco_system_captions, no_spice=args.no_spice)
    coco_eval.params['image_id'] = coco_system_captions.getImgIds()

    coco_eval.evaluate()

    print('\nScores:')
    print('=======')
    for metric, score in coco_eval.eval.items():
        print('{}: {:.5f}'.format(metric, score))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-captions', type=lambda x: is_valid_file(parser, x, ['json']), required=True)
    parser.add_argument('--system-captions', type=lambda x: is_valid_file(parser, x, ['json']), required=True)
    parser.add_argument('--no-spice', action='store_true', default=False)
    return parser


def is_valid_file(parser, arg, file_types):
    ext = re.sub(r'^\.', '', os.path.splitext(arg)[1])

    if not os.path.exists(arg):
        parser.error('File not found: "{}"'.format(arg))
    elif ext not in file_types:
        parser.error('Invalid "{}" provided. Only files of type {} are allowed'.format(arg, file_types))
    else:
        return arg


if __name__ == '__main__':
    main()
