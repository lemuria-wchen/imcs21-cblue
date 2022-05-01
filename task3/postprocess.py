import json
import os
import re
import argparse


KEYS = ['主诉', '现病史', '辅助检查', '既往史', '诊断', '建议']

PATTERNS = [
    re.compile('主诉：(.*)现病史：'),
    re.compile('现病史：(.*)辅助检查：'),
    re.compile('辅助检查：(.*)既往史：'),
    re.compile('既往史：(.*)诊断：'),
    re.compile('诊断：(.*)建议：'),
    re.compile('建议：(.*)'),
]


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json(data, path: str, indent=4):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def grep(text):
    report = {}
    for key, pattern in zip(KEYS, PATTERNS):
        results = pattern.findall(text)
        if len(results) > 0:
            report[key] = results[0]
        else:
            report[key] = ''
    return report


def load_preds(pred_path, target='no_t5'):
    preds = []
    if target == 't5':
        with open(pred_path, 'r') as f:
            for line in f.readlines():
                pred, _ = line.split('\t')
                preds.append(pred)
    else:
        with open(pred_path, 'r') as f:
            for line in f.readlines():
                preds.append(''.join(line.strip().split()))
    return preds


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_path', type=str, default='../../dataset/test_input.json', help='gold file path')
    parser.add_argument('--pred_path', type=str, default='pred_lstm.txt', help='gold file path')
    parser.add_argument('--target', type=str, default='no_t5')

    args = parser.parse_args()
    if args.target == 't5':
        save_path = args.pred_path.replace('.tsv', '.json')
    else:
        save_path = args.pred_path.replace('.txt', '.json')

    test_input = load_json(args.gold_path)
    pred_texts = load_preds(args.pred_path, args.target)

    final = {}
    for i, (pid, _) in enumerate(test_input.items()):
        final[pid] = grep(pred_texts[i])

    write_json(final, path=save_path)

    # usage:
    # python postprocess.py --gold_path ../../dataset/test_input.json --pred_path opennmt/data/pred_lstm.txt --target no_t5
