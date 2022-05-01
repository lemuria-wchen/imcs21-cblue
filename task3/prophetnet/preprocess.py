from transformers import BertTokenizer
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

tok = BertTokenizer.from_pretrained('bert-base-chinese')


def process(title):
    x = []
    for key, value in title.items():
        x.append(key + '：' + value)
    return ''.join(x)


def make_data(samples, mode='train', max_src_len=500, max_tgt_len=240):
    src, tgt = '', ''
    i = 0
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
        content = " ".join(tok.tokenize(content.strip())[: max_src_len])
        title1 = " ".join(tok.tokenize(title1.strip())[: max_tgt_len])
        title2 = " ".join(tok.tokenize(title2.strip())[: max_tgt_len])
        if mode == 'train':
            src += content + '\n' + content + '\n'
            tgt += title1 + '\n' + title2 + '\n'
        else:
            src += content + '\n'
            tgt += title1 + '\n'
    with open(os.path.join('data', 'tokenized_{}.src'.format(mode)), 'w', encoding='utf-8') as f:
        f.write(src)
    with open(os.path.join('data', 'tokenized_{}.tgt'.format(mode)), 'w', encoding='utf-8') as f:
        f.write(tgt)
    if mode == 'test':
        assert i == len(test_prediction)


make_data(train_set, mode='train')
make_data(dev_set, mode='dev')
make_data(test_set, mode='test')
