import json
import argparse
from rouge import Rouge


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def process(title, delimiter=''):
    x = []
    for key, value in title.items():
        x.append(key + '：' + value)
    return delimiter.join(x)


def compute_rouge(source, targets):
    try:
        r1, r2, rl = 0, 0, 0
        n = len(targets)
        for target in targets:
            source, target = ' '.join(source), ' '.join(target)
            scores = Rouge().get_scores(hyps=source, refs=target)
            r1 += scores[0]['rouge-1']['f']
            r2 += scores[0]['rouge-2']['f']
            rl += scores[0]['rouge-l']['f']
        return {
            'rouge-1': r1 / n,
            'rouge-2': r2 / n,
            'rouge-l': rl / n,
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }


def compute_rouges(sources, targets):
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
    }
    for source, target in zip(sources, targets):
        score = compute_rouge(source, target)
        for k, v in scores.items():
            scores[k] = v + score[k]
    results = {k: v / len(targets) for k, v in scores.items()}
    print(results)
    # 在 CBLEU 评测中，我们使用 rouge-1, rouge-2 和 rouge-l 的平均值作为评价指标
    return (results['rouge-1'] + results['rouge-2'] + results['rouge-l']) / 3


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_path', type=str, default='../../dataset/test.json', help='gold file path')
    parser.add_argument('--pred_path', type=str, help='pred file path')

    args = parser.parse_args()

    gold_data = load_json(args.gold_path)
    pred_data = load_json(args.pred_path)

    golds, preds = [], []
    for pid, sample in gold_data.items():
        title1, title2 = sample['report'][0], sample['report'][1]
        golds.append([process(title1), process(title2)])
        assert pid in pred_data
        preds.append(process(pred_data[pid]))

    print('-- MRG task evaluation --')
    compute_rouges(preds, golds)

    # usage:
    # python eval_task3.py --gold_path ../../dataset/test.json --pred_path opennmt/data/pred_lstm.json
