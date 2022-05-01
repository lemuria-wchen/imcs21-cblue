import json
import os
import numpy as np


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


data_dir = '../../../dataset'
train_set = load_json(os.path.join(data_dir, 'train.json'))
dev_set = load_json(os.path.join(data_dir, 'dev.json'))
test_set = load_json(os.path.join(data_dir, 'test.json'))

save_path = '../ernie_predictions.npz'
test_prediction = np.load(save_path)['test_prediction']


def process(title):
    x = []
    for key, value in title.items():
        x.append(key + '：' + value)
    return ''.join(x)


def make_data(samples, path, mode='train'):
    lines = ''
    i = 0
    with open(path, 'w', encoding='utf-8') as f:
        for pid, sample in samples.items():
            content = []
            if mode == 'test':
                for sent in sample['dialogue']:
                    if test_prediction[i] != 15:
                        content.append(sent['speaker'] + sent['sentence'])
                    i += 1
            else:
                for sent in sample['dialogue']:
                    if sent['dialogue_act'] != 'Other':
                        content.append(sent['speaker'] + sent['sentence'])
            content = ''.join(content)
            title1, title2 = process(sample['report'][0]), process(sample['report'][1])
            if mode == 'train':
                lines += title1 + '\t' + content + '\n'
                lines += title2 + '\t' + content + '\n'
            elif mode == 'dev':
                lines += title1 + '\t' + content + '\n'
            elif mode == 'dev_for_test':
                lines += title1 + '\t' + content + '\n'
            else:
                lines += content + '\n'
        f.write(lines)
    if mode == 'test':
        assert i == len(test_prediction)


make_data(train_set, 'data/train.tsv', mode='train')
make_data(dev_set, 'data/dev.tsv', mode='dev')
make_data(dev_set, 'data/dev_predict.tsv', mode='dev_for_test')
make_data(test_set, 'data/predict.tsv', mode='test')
