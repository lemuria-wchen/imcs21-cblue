import json
from sklearn import metrics
import argparse


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def dac_eval(gold_data, pred_data, digits=4):
    eids = list(gold_data.keys())
    gold_das = []
    pred_das = []
    for eid in eids:
        gold_dialogue = gold_data[eid]["dialogue"]
        for sent in gold_dialogue:
            sid = sent['sentence_id']
            gold_das.append(sent['dialogue_act'])
            pred_das.append(pred_data[eid][sid])
    p = metrics.precision_score(gold_das, pred_das, average='macro')
    r = metrics.recall_score(gold_das, pred_das, average='macro')
    f1 = metrics.f1_score(gold_das, pred_das, average='macro')
    acc = metrics.accuracy_score(gold_das, pred_das)
    print('P: {}, R: {}, F1: {}, Acc: {}'.format(
        round(p, digits), round(r, digits), round(f1, digits), round(acc, digits)))
    # 在 CBLEU 评测中，我们以准确率作为评价指标 (Acc)
    return acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_path', type=str, default='../../dataset/test.json', help='gold file path')
    parser.add_argument('--pred_path', type=str, help='pred file path')
    args = parser.parse_args()

    grounds = load_json(args.gold_path)
    predictions = load_json(args.pred_path)

    print('-- DAC task evaluation --')
    dac_eval(grounds, predictions)
