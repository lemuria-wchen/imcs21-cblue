# -*- coding:utf-8 -*-
import json
import sys
from seqeval.metrics import f1_score

def load_json(path: str):
    '''读取json文件'''
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def eval(gold_data, pred_data):
    """评估F1值"""
    golds = []
    preds = []
    eids = list(gold_data.keys())
    for eid in eids:
        gold_dialogue = gold_data[eid]["dialogue"]
        for sent in gold_dialogue:
            sid = sent['sentence_id']
            gold_bio = sent['BIO_label'].split(' ')
            pred_bio = pred_data[eid][sid].split(' ')
            assert len(gold_bio) == len(pred_bio)
            golds.append(gold_bio)
            preds.append(pred_bio)
    assert len(golds) == len(preds)
    f1 = f1_score(golds, preds)
    print('Test F1 score {}%'.format(round(f1 * 100, 4)))

if __name__ == "__main__":

    gold_data = load_json(sys.argv[1])
    pred_data = load_json(sys.argv[2])

    eval(gold_data, pred_data)
