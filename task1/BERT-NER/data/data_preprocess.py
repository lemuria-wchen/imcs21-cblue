# -*- coding:utf-8 -*-
import pandas as pd
import json
import os


def read_train_data(fn):
    """读取用于训练的json数据"""
    with open(fn, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    return data

def read_test_data(fn):
    """读取用于测试的json数据"""
    with open(fn, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    return data

def read_example_ids(fn):
    """读取划分数据集的文件"""
    example_ids = pd.read_csv(fn)
    return example_ids

def save_train_data(data, example_ids, mode, fn1, fn2):
    """
    训练集和验证集的数据转换
    :param data: 用于训练的json数据
    :param example_ids: 样本id划分数据
    :param mode: train/dev
    :param fn1: 文本序列 input.seq.char
    :param fn2: BIO序列标签 output.seq.bio
    :return:
    """
    eids = example_ids[example_ids['split'] == mode]['example_id'].to_list()
    # # sample
    # n = len(eids)
    # eids = eids[:int(n*0.01)]
    seq_in, seq_bio = [], []
    for eid in eids:
        tmp_data = data[str(eid)]
        tmp_dialogue = tmp_data['dialogue']
        for i in range(len(tmp_dialogue)):
            tmp_sent = list(tmp_dialogue[i]['speaker'] + '：' + tmp_dialogue[i]['sentence'])
            tmp_bio = ['O'] * 3 + tmp_dialogue[i]['BIO_label'].split(' ')
            assert len(tmp_sent) == len(tmp_bio)
            seq_in.append(tmp_sent)
            seq_bio.append(tmp_bio)
    assert len(seq_in) == len(seq_bio)
    print(mode, '句子数量为：', len(seq_in))
    # 数据保存
    with open(fn1, 'w', encoding='utf-8') as f1:
        for i in seq_in:
            tmp = ' '.join(i)
            f1.write(tmp+'\n')
    with open(fn2, 'w', encoding='utf-8') as f2:
        for i in seq_bio:
            tmp = ' '.join(i)
            f2.write(tmp+'\n')

def get_vocab_char(fr1, fr2, fw):
    """获得字符种类字典"""
    chars = []
    with open(fr1, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            for i in line:
                if i not in chars:
                    chars.append(i)
    with open(fr2, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            for i in line:
                if i not in chars:
                    chars.append(i)
    add_tokens = ['[PAD]', '[UNK]', '[SEP]', '[SPA]']
    chars = add_tokens + chars
    print('字符种类：', len(chars))

    with open(fw, 'w', encoding='utf-8') as f:
        for w in chars:
            f.write(w + '\n')

def get_vocab_bio(fr1, fr2, fw):
    """获得bio种类字典"""
    bio = []
    with open(fr1, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            for i in line:
                if i not in bio:
                    bio.append(i)
    with open(fr2, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            for i in line:
                if i not in bio:
                    bio.append(i)

    bio = sorted(list(bio), key=lambda x: (x[2:], x[:2]))
    add_tokens = ["PAD", "UNK"]
    bio = add_tokens + bio
    print('bio种类：', len(bio))

    with open(fw, 'w', encoding='utf-8') as f:
        for w in bio:
            f.write(w + '\n')


if __name__ == "__main__":

    train_data = read_train_data('../../../dataset/train.json')
    example_ids = read_example_ids('../../../dataset/split.csv')

    data_dir = 'ner_data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.makedirs(data_dir+'/train')
        os.makedirs(data_dir+'/dev')

    # 获得训练数据
    save_train_data(
        train_data,
        example_ids,
        'train',
        os.path.join(data_dir, 'train', 'input.seq.char'),
        os.path.join(data_dir, 'train', 'output.seq.bio')
    )

    # 获得验证数据
    save_train_data(
        train_data,
        example_ids,
        'dev',
        os.path.join(data_dir, 'dev', 'input.seq.char'),
        os.path.join(data_dir, 'dev', 'output.seq.bio')
    )

    # 获取一些vocab信息
    get_vocab_bio(
        os.path.join(data_dir, 'train', 'output.seq.bio'),
        os.path.join(data_dir, 'dev', 'output.seq.bio'),
        os.path.join(data_dir, 'vocab_bio.txt')
    )
