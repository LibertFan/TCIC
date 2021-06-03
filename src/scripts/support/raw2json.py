import json
from argparse import ArgumentParser


def set_options():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    return args


def convert(args):

    img_id_to_caption_list = []

    with open(args.input, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if "H-" in line:
                raw_img_id, _, hypo = line.split('\t')
                img_id = raw_img_id.split('-')[-1]
                if not hypo.endswith('.'):
                    hypo = hypo+' .'
                img_id_to_caption = {'caption': hypo, 'image_id': int(img_id)}
                img_id_to_caption_list.append(img_id_to_caption)
    f.close()

    with open(args.output, 'w') as f:
        json.dump(img_id_to_caption_list, f)
    f.close()


def main():
    args = set_options()
    convert(args)


if __name__ == "__main__":
    main()
