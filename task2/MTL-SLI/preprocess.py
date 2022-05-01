import os
import pandas as pd
from utils import load_json, get_entity_bio, write_json_by_line


def make_dataset(samples, num_contexts=2, mode='train', delimiter='\t'):
    instances = []
    for pid, sample in samples.items():
        diag = sample['dialogue']
        num_utts = len(diag)
        for i in range(num_contexts,  num_utts - num_contexts):
            utts, bio_tags, s_names, s_labels = [], [], [], []
            for j in range(i - num_contexts, i + num_contexts + 1):
                utt = diag[j]['speaker'] + '：' + diag[j]['sentence']
                if mode != 'test':
                    s_names.extend(diag[j]['symptom_norm'])
                    s_labels.extend(diag[j]['symptom_type'])
                    bio_tag = ['O'] * 3 + diag[j]['BIO_label'].split()
                    assert len(utt) == len(bio_tag)
                else:
                    bio_tag = []
                utts.append(utt)
                bio_tags.append(bio_tag)
            utts_c, bio_tags_c, bio_ids_c, chunks_c = delimiter.join(utts), [], [], []
            if mode != 'test':
                for tags in bio_tags:
                    bio_tags_c.extend(tags)
                    bio_tags_c.extend(['O'] * len(delimiter))
                for _ in range(len(delimiter)):
                    bio_tags_c.pop(-1)
                assert len(utts_c) == len(bio_tags_c)
                chunks = [chunk for chunk in get_entity_bio(bio_tags_c) if chunk[0] == 'Symptom']
                for chunk, sn, st in zip(chunks, s_names, s_labels):
                    assert sn in sym2id
                    assert st in sl2id
                    # task 2/3: predict normalized name and symptom label
                    chunks_c.append([chunk[1], chunk[2], sym2id.get(sn), sl2id.get(st)])
                bio_ids_c = [bio2id.get(tag) for tag in bio_tags_c]
            instance = {
                'pid': pid,
                'utts': utts_c,
                'bio_ids': bio_ids_c,
                'chunks': chunks_c,
            }
            instances.append(instance)
    return instances


if __name__ == '__main__':

    data_dir = '../../../dataset'
    # data_dir = 'dataset'
    train_set = load_json(os.path.join(data_dir, 'train.json'))
    dev_set = load_json(os.path.join(data_dir, 'dev.json'))
    test_set = load_json(os.path.join(data_dir, 'test_input.json'))

    saved_path = 'data/c2'
    os.makedirs(os.path.join(saved_path), exist_ok=True)

    sym2id = {value: key for key, value in pd.read_csv(os.path.join(data_dir, 'symptom_norm.csv'))['norm'].items()}
    bio2id = {item: idx for idx, item in enumerate([
        'PAD', 'O', 'B-Symptom', 'I-Symptom', 'B-Medical_Examination', 'I-Medical_Examination',
        'B-Drug', 'I-Drug', 'B-Drug_Category', 'I-Drug_Category', 'B-Operation', 'I-Operation']
    )}
    sl2id = {'0': 0, '1': 1, '2': 2}

    train = make_dataset(train_set, mode='train')
    dev = make_dataset(dev_set, mode='dev')
    test = make_dataset(test_set, mode='test')

    print('train/dev/test size: {}/{}/{}'.format(len(train), len(dev), len(test)))

    write_json_by_line(train, os.path.join(saved_path, 'train.json'))
    write_json_by_line(dev, os.path.join(saved_path, 'dev.json'))
    write_json_by_line(test, os.path.join(saved_path, 'test.json'))
