import json
from collections import Counter
from seqeval.metrics import precision_score, recall_score, f1_score
import argparse


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_entity_bio(seq):
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def compute(origin, found, right):
    recall = 0 if origin == 0 else (right / origin)
    precision = 0 if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    return recall, precision, f1


def ner_eval(gold_data, pred_data):
    eids = list(gold_data.keys())
    origins = []
    founds = []
    rights = []
    golds = []
    preds = []
    for eid in eids:
        gold_dialogue = gold_data[eid]["dialogue"]
        for sent in gold_dialogue:
            sid = sent['sentence_id']
            gold_bio = sent['BIO_label'].split(' ')
            pred_bio = pred_data[eid][sid].split(' ')
            assert len(gold_bio) == len(pred_bio)
            label_entities = get_entity_bio(gold_bio)
            pre_entities = get_entity_bio(pred_bio)
            origins.extend(label_entities)
            founds.extend(pre_entities)
            rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])
            golds.append(gold_bio)
            preds.append(pred_bio)
    class_info = {}
    origin_counter = Counter([x[0] for x in origins])
    found_counter = Counter([x[0] for x in founds])
    right_counter = Counter([x[0] for x in rights])
    for type_, count in origin_counter.items():
        origin = count
        found = found_counter.get(type_, 0)
        right = right_counter.get(type_, 0)
        recall, precision, f1 = compute(origin, found, right)
        class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
    origin = len(origins)
    found = len(founds)
    right = len(rights)
    _, _, overall_f1 = compute(origin, found, right)
    # entity-level metric
    sx_f1 = class_info['Symptom']['f1']
    dn_f1 = class_info['Drug']['f1']
    dc_f1 = class_info['Drug_Category']['f1']
    ex_f1 = class_info['Medical_Examination']['f1']
    op_f1 = class_info['Operation']['f1']
    print('-'*20, 'entity-level metric (f1 score)', '-'*20)
    print('SX: {}\tDN: {}\tDC: {}\tEX: {}\tOP: {}\tOverall: {}'.format(
        round(sx_f1, 4), round(dn_f1, 4), round(dc_f1, 4), round(ex_f1, 4), round(op_f1, 4), round(overall_f1, 4),
    ))
    # token-level metric
    token_p = precision_score(golds, preds)
    token_r = recall_score(golds, preds)
    token_f1 = f1_score(golds, preds)
    print('-'*20, 'token-level metric', '-'*20)
    print('P: {}\tR: {}\tF: {}'.format(
        round(token_p, 4), round(token_r, 4), round(token_f1, 4),
    ))
    # 在 CBLEU 评测中，我们以实体级 F1 score 作为评价指标（overall_f1）
    # return sx_f1, dn_f1, dc_f1, ex_f1, op_f1, overall_f1, token_p, token_r, token_f1
    return overall_f1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_path', type=str, default='../../dataset/test.json', help='gold file path')
    parser.add_argument('--pred_path', type=str, help='pred file path')
    args = parser.parse_args()

    grounds = load_json(args.gold_path)
    predictions = load_json(args.pred_path)

    print('-- NER task evaluation --')
    ner_eval(grounds, predictions)
