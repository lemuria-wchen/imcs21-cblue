import json
import os
import numpy as np


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


data_dir = '../../../dataset'
# data_dir = 'dataset'
train_set = load_json(os.path.join(data_dir, 'train.json'))
dev_set = load_json(os.path.join(data_dir, 'dev.json'))
# you can just import the test input file and replace the target with random strings in `make_data` function
test_set = load_json(os.path.join(data_dir, 'test.json'))


save_path = '../ernie_predictions.npz'
test_prediction = np.load(save_path)['test_prediction']


def process(title):
    x = []
    for key, value in title.items():
        x.append(key + '：' + value)
    return '\t'.join(x)


def make_data(samples, mode='train'):
    src, tgt = '', ''
    src_len = []
    tgt_len = []
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
        src_len.append(len(content))
        content = ' '.join(content)
        title1, title2 = process(sample['report'][0]), process(sample['report'][1])
        tgt_len.append(len(title1))
        tgt_len.append(len(title2))
        title1 = ' '.join(title1)
        title2 = ' '.join(title2)
        if mode == 'train':
            src += content + '\n' + content + '\n'
            tgt += title1 + '\n' + title2 + '\n'
        else:
            src += content + '\n'
            tgt += title1 + '\n'    # whatever text is ok

    print('src len, avg / 95p / max: {} / {} / {}'.format(np.round(np.mean(src_len), 2), np.quantile(src_len, 0.95), np.max(src_len)))
    print('tgt len, avg / 95p / max: {} / {} / {}'.format(np.round(np.mean(tgt_len), 2), np.quantile(tgt_len, 0.95), np.max(tgt_len)))
    with open(os.path.join('data', 'src-{}.txt'.format(mode)), 'w', encoding='utf-8') as f:
        f.write(src)
    with open(os.path.join('data', 'tgt-{}.txt'.format(mode)), 'w', encoding='utf-8') as f:
        f.write(tgt)
    if mode == 'test':
        assert i == len(test_prediction)


make_data(train_set, mode='train')
make_data(dev_set, mode='dev')
make_data(test_set, mode='test')
