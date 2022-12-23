import json
import os


def load_json(fn):
    with open(fn, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    return data


def save_data(samples, input_fn, output_fn2):
    seq_in = []
    seq_bio = []
    for pid, sample in samples.items():
        for item in sample['dialogue']:
            sent = list(item['speaker'] + '：' + item['sentence'])
            bio = ['O'] * 3 + item['BIO_label'].split(' ')
            # bio = ['O'] * 3 + item['new_BIO_label'].split(' ')
            assert len(sent) == len(bio)
            seq_in.append(sent)
            seq_bio.append(bio)
    assert len(seq_in) == len(seq_bio)
    print('句子数量为：', len(seq_in))
    with open(input_fn, 'w', encoding='utf-8') as f1:
        for i in seq_in:
            tmp = ' '.join(i)
            f1.write(tmp + '\n')
    with open(output_fn2, 'w', encoding='utf-8') as f2:
        for i in seq_bio:
            tmp = ' '.join(i)
            f2.write(tmp + '\n')


def get_vocab_char(fr1, fr2, fw):
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
    print('bio种类：', len(bio))

    with open(fw, 'w', encoding='utf-8') as f:
        for w in bio:
            f.write(w + '\n')


if __name__ == "__main__":

    data_dir = '../../../dataset'
    # data_dir = 'dataset'
    train_set = load_json(os.path.join(data_dir, 'V2/train_update.json'))
    dev_set = load_json(os.path.join(data_dir, 'V2/train_update.json'))

    saved_dir = 'ner_data'
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
        os.makedirs(os.path.join(saved_dir, 'train'))
        os.makedirs(os.path.join(saved_dir, 'dev'))

    # 获得训练数据
    save_data(
        train_set,
        os.path.join(saved_dir, 'train', 'input.seq.char'),
        os.path.join(saved_dir, 'train', 'output.seq.bio')
    )

    # 获得验证数据
    save_data(
        dev_set,
        os.path.join(saved_dir, 'dev', 'input.seq.char'),
        os.path.join(saved_dir, 'dev', 'output.seq.bio')
    )

    # 获取一些vocab信息
    get_vocab_bio(
        os.path.join(saved_dir, 'train', 'output.seq.bio'),
        os.path.join(saved_dir, 'dev', 'output.seq.bio'),
        os.path.join(saved_dir, 'vocab_bio.txt')
    )

    get_vocab_char(
        os.path.join(saved_dir, 'train', 'input.seq.char'),
        os.path.join(saved_dir, 'dev', 'input.seq.char'),
        os.path.join(saved_dir, 'vocab_char.txt')
    )
