# -*- coding:utf-8 -*-
import json
import sys

def load_json(path: str):
    '''读取json文件'''
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def cal_f1_score(preds, golds):
    """样本级别的症状识别评价方式"""
    assert len(preds) == len(golds)
    p_sum = 0
    r_sum = 0
    hits = 0
    for pred, gold in zip(preds, golds):
        p_sum += len(pred)
        r_sum += len(gold)
        for k, v in pred.items():
            if k in gold and v == gold[k]:
                hits += 1
    p = hits / p_sum if p_sum > 0 else 0
    r = hits / r_sum if r_sum > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1

def eval(gold_data, pred_data):
    """评估F1值"""
    assert len(gold_data) == len(pred_data)
    golds = []
    preds = []
    eids = list(gold_data.keys())
    for eid in eids:
        gold_type = gold_data[eid]['implicit_info']['Symptom']
        pred_type = pred_data[eid]
        golds.append(gold_type)
        preds.append(pred_type)
    assert len(golds) == len(preds)
    _, _, f1 = cal_f1_score(preds, golds)
    print('Test F1 score {}%'.format(round(f1 * 100, 4)))

if __name__ == "__main__":

    gold_data = load_json(sys.argv[1])  # 读入test的真实数据
    pred_data = load_json(sys.argv[2])  # 读入test的预测数据

    eval(gold_data, pred_data)
