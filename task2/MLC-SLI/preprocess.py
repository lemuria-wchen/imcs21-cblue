import os
import numpy as np
from utils import load_json, write_json_by_line
import argparse


def make_label_exp_v1(symptoms):
    label = [0] * num_labels
    for symptom in symptoms:
        if sym2id.get(symptom) is not None:
            label[sym2id.get(symptom)] = 1
        else:
            raise Exception
    return label


def make_label_exp(symptoms):
    label = []
    for sn in symptoms:
        if sn in sym2id:
            label.append(sym2id.get(sn))
    return label


def make_dataset_exp_v1(samples, weight=5, mode='train', add_imp=False):
    instances, input_lens = [], []
    for pid, sample in samples.items():
        y = make_label_exp(sample['explicit_info']['Symptom']) if 'diagnosis' in sample else []
        if mode == 'train' and add_imp:
            for utt in sample['dialogue']:
                if 'symptom_norm' in utt and len(utt['symptom_norm']) > 0:
                    instances.append((utt['sentence'], y, pid, 1))
        input_lens.append(len(sample['self_report']))
        instances.append((sample['self_report'], y, pid, weight))
    return instances, input_lens


def make_dataset_exp(samples, weight):
    instances = []
    for pid, sample in samples.items():
        utts = '患者：' + sample['self_report']
        label = make_label_exp(sample['explicit_info']['Symptom']) if 'diagnosis' in sample else []
        instances.append({
            'pid': pid,
            'utts': utts,
            'label': label,
            'weight': weight,
        })
    return instances


def make_label_imp_v1(label, symptom_norm, symptom_type):
    assert len(symptom_norm) == len(symptom_type)
    for sx_norm, sx_type in zip(symptom_norm, symptom_type):
        assert sx_norm in sym2id
        label[sym2id.get(sx_norm) * len(sl2id) + int(sx_type)] = 1


def make_label_imp_v2(symptom_norm, symptom_type):
    assert len(symptom_norm) == len(symptom_type)
    label = []
    for sx_norm, sx_type in zip(symptom_norm, symptom_type):
        assert sx_norm in sym2id
        label.append(sym2id.get(sx_norm) * len(sl2id) + int(sx_type))
    return label


def make_label_imp(symptoms):
    label = []
    for sn, sl in symptoms.items():
        if sn in sym2id:
            label.append(sym2id.get(sn) * len(sl2id) + sl2id.get(sl))
    return label


def make_dataset_imp(samples, num_contexts, weight, mode='train', delimiter='｜'):
    instances = []
    pos, neg = 0, 0
    for pid, sample in samples.items():
        # break
        diag = sample['dialogue']
        num_utts = len(diag)
        for i in range(num_utts):
            utts, label = [], []
            has_symptom = False
            for j in range(max(0, i - num_contexts), min(i + num_contexts + 1, num_utts)):
                utts.append(diag[j]['speaker'] + '：' + diag[j]['sentence'])
                if mode != 'test':
                    label = make_label_imp(diag[i]['local_implicit_info'])
                    if len(diag[i]['local_implicit_info']) > 0:
                        has_symptom = True
            utts = delimiter.join(utts)
            if has_symptom:
                instance_weight = weight
                pos += 1
            else:
                neg += 1
                instance_weight = 1
            instances.append({
                'pid': pid,
                'sid': diag[i]['sentence_id'],
                'utts': utts,
                'label': label,
                'weight': instance_weight,
            })
    print('proportion of utterances containing symptom entities: {} %'.format(
        round(100 * pos / (pos + neg), 2)))
    return instances


if __name__ == '__main__':

    data_dir = '../../../dataset'
    # data_dir = 'dataset'

    _num_contexts = 0
    _weight = 20

    train_set = load_json(os.path.join(data_dir, 'V3/train_update.json'))
    dev_set = load_json(os.path.join(data_dir, 'V3/dev_update.json'))
    test_set = load_json(os.path.join(data_dir, 'V3/test_update.json'))

    # from collections import defaultdict
    # tmp = set()
    # for pid, sample in train_set.items():
    #     for s in sample['explicit_info']['Symptom']:
    #         tmp.add(s)
    #     for sent in sample['dialogue']:
    #         for key, val in zip(sent['symptom_norm'], sent['symptom_type']):
    #             tmp.add(key)
    # for pid, sample in dev_set.items():
    #     for s in sample['explicit_info']['Symptom']:
    #         tmp.add(s)
    #     for sent in sample['dialogue']:
    #         for key, val in zip(sent['symptom_norm'], sent['symptom_type']):
    #             tmp.add(key)
    # for pid, sample in test_set.items():
    #     for s in sample['explicit_info']['Symptom']:
    #         tmp.add(s)
    #     for sent in sample['dialogue']:
    #         for key, val in zip(sent['symptom_norm'], sent['symptom_type']):
    #             tmp.add(key)

    # # 验证
    # for pid, sample in train_set.items():
    #     tmp = defaultdict(list)
    #     for sent in sample['dialogue']:
    #         for key, val in zip(sent['new_symptom_norm'], sent['new_symptom_type']):
    #             tmp[key].append(val)
    #     print(tmp)
    #     for key, val in tmp.items():
    #         if len(val) > 1:
    #             assert len(set(val)) == 1
    #     assert len(tmp) == len(sample['implicit_info']['Symptom'])

    # load normalized symptom
    # sym2id = {value: key for key, value in pd.read_csv(os.path.join(data_dir, 'V2/symptom_norm.csv'))['norm'].items()}
    # num_labels = len(sym2id)

    mappings_path = os.path.join(data_dir, 'V3/mappings.json')
    sym2id, _, _, _, sl2id, _ = load_json(mappings_path)
    num_labels = len(sym2id)

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default='exp', required=True, type=str)
    args = parser.parse_args()

    saved_path = 'data'
    suffix = args.target
    os.makedirs(os.path.join(saved_path, suffix), exist_ok=True)

    if args.target == 'exp':
        train = make_dataset_exp(train_set, weight=_weight)
        dev = make_dataset_exp(dev_set, weight=_weight)
        test = make_dataset_exp(test_set, weight=_weight)
        print('train/dev/test size: {}/{}/{}'.format(len(train), len(dev), len(test)))
    else:
        train = make_dataset_imp(train_set, num_contexts=_num_contexts, weight=_weight, mode='train')
        dev = make_dataset_imp(dev_set, num_contexts=_num_contexts, weight=_weight,  mode='dev')
        test = make_dataset_imp(test_set, num_contexts=_num_contexts, weight=_weight,  mode='test')
        print('train/dev/test size: {}/{}/{}'.format(len(train), len(dev), len(test)))

    lens = np.array([len(sample['utts']) for sample in train])
    print('train/dev/test size: {}/{}/{}'.format(len(train), len(dev), len(test)))
    print('avg. utts lens: {}, 95 % utts lens: {}'.format(
        np.round(np.mean(lens), 4), np.round(np.percentile(lens, 99), 4)))

    write_json_by_line(train, os.path.join(saved_path, suffix, 'train.json'))
    write_json_by_line(dev, os.path.join(saved_path, suffix, 'dev.json'))
    write_json_by_line(test, os.path.join(saved_path, suffix, 'test.json'))
