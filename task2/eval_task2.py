import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

import argparse


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def make_label(symptoms, target):
    if target == 'exp':
        label = [0] * num_labels
        for sx in symptoms:
            if sym2id.get(sx) is not None:
                label[sym2id.get(sx)] = 1
    else:
        label = [0] * (num_labels * 3)
        for sx_norm, sx_type in symptoms.items():
            if sym2id.get(sx_norm) is not None:
                label[sym2id.get(sx_norm) * 3 + int(sx_type)] = 1
    return label


def hamming_score(
    s, preds):
    assert len(golds) == len(preds)
    out = np.ones(len(golds))
    n = np.logical_and(golds, preds).sum(axis=1)
    d = np.logical_or(golds, preds).sum(axis=1)
    return np.mean(np.divide(n, d, out=out, where=d != 0))


def multi_label_metric(golds, preds):
    # Example-based Metrics
    print('Exact Match Ratio: {}'.format(accuracy_score(golds, preds, normalize=True, sample_weight=None)))
    print('Hamming loss: {}'.format(hamming_loss(golds, preds)))
    print('Hamming score: {}'.format(hamming_score(golds, preds)))
    # Label-based Metrics
    print('micro Recall: {}'.format(recall_score(y_true=golds, y_pred=preds, average='micro', zero_division=0)))
    print('micro Precision: {}'.format(precision_score(y_true=golds, y_pred=preds, average='micro', zero_division=0)))
    print('micro F1: {}'.format(f1_score(y_true=golds, y_pred=preds, average='micro', zero_division=0)))
    # 在 CBLEU 评测中，我们标签级 F1 score 作为评价指标（micro F1）
    return f1_score(y_true=golds, y_pred=preds, average='micro', zero_division=0)


def labels_metric(golds, preds):
    f1 = f1_score(golds, preds, average='macro')
    acc = accuracy_score(golds, preds)
    labels = classification_report(golds, preds, output_dict=True)
    print('F1-score -> positive: {}, negative: {}, uncertain: {}, overall: {}, acc: {}'.format(
        labels['1']['f1-score'], labels['0']['f1-score'], labels['2']['f1-score'], f1, acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_path", required=True, type=str)
    parser.add_argument("--pred_path", required=True, type=str)
    args = parser.parse_args()

    gold_data = load_json(args.gold_path)
    pred_data = load_json(args.pred_path)

    # load normalized symptom（需要导入标准化症状）
    # sn_path = 'symptom_norm.csv'
    # sym2id = {value: key for key, value in pd.read_csv(sn_path)['norm'].items()}
    # num_labels = len(sym2id)
    
    # load normalized symptom
    mappings_path = 'mappings.json'
    sym2id, _, _, _, _, _ = load_json(mappings_path)
    num_labels = len(sym2id)

    print('---- dialogue level ----')
    golds_full, preds_full = [], []
    # gold_labels, pred_labels = [], []
    for pid, sample in gold_data.items():
        gold = sample['implicit_info']['Symptom']
        pred = pred_data.get(pid)['global']
        golds_full.append(make_label(gold, target='imp'))
        preds_full.append(make_label(pred, target='imp'))

    golds_full, preds_full = np.array(golds_full), np.array(preds_full)
    print('-- SR task evaluation --')
    multi_label_metric(golds_full, preds_full)
    
    print('---- utterance level ----')
    # utterance level
    golds_full, preds_full = [], []
    gold_labels, pred_labels = [], []
    for pid, sample in gold_data.items():
        for sent in sample['dialogue']:
            sid = sent['sentence_id']
            gold = sent['local_implicit_info']
            pred = pred_data.get(pid).get(sid)
            golds_full.append(make_label(gold, target='imp'))
            preds_full.append(make_label(pred, target='imp'))

    golds_full, preds_full = np.array(golds_full), np.array(preds_full)
    print('-- SR task evaluation --')
    multi_label_metric(golds_full, preds_full)
