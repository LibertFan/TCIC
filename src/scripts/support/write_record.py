import argparse
import collections
import json
import os
import json


def get_result(result_txt):
    scores = dict()
    with open(result_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if ('Bleu' in line or 'METEOR' in line or 'ROUGE_L' in line
                    or 'CIDEr' in line) and ': ' in line and len(line.strip().split()) == 2:
                try:
                    critic, score = line.strip().split(': ')
                    scores[critic] = float(score)
                except:
                    return -1
    f.close()

    if len(scores) < 7:
        return -1
    return scores


def main():
    parser = argparse.ArgumentParser(
        description='Write the records of different best models to a whole xlsx file',
    )
    parser.add_argument('--result_path', type=str, help='The path of the model result')
    parser.add_argument('--save_path', type=str, help='The path of the model result')
    parser.add_argument('--ckpt', type=str, help='checkpoint_name')
    parser.add_argument('--beam', type=str, help='beam size')
    parser.add_argument('--min_len', type=str, help='min length')
    parser.add_argument('--max_len', type=str, help='max length')
    parser.add_argument('--ngram_size', type=str, help='ngram size')
    parser.add_argument('--len_pen', type=str, help='length penalty')

    args = parser.parse_args()
    print(args)

    result = get_result(args.result_path)
    if isinstance(result, dict):
        result['ckpt'] = args.ckpt
        result['beam'] = args.beam
        result['min_len'] = args.min_len
        result['max_len'] = args.max_len
        result['ngram_size'] = args.ngram_size
        result['len_pen'] = args.len_pen

        with open(args.save_path, 'w') as fw:
            json.dump(result, fw)
        fw.close()



if __name__ == '__main__':
    main()
