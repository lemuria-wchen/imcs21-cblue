import os
import numpy as np
import pandas as pd
from utils import load_json, write_json, get_entity_bio, write_json_by_line
from tqdm import tqdm


def make_dataset(samples, num_contexts=1, mode='train', delimiter='｜'):
    instances = []
    pos, neg = 0, 0
    for pid, sample in tqdm(samples.items()):
        # break
        diag = sample['dialogue']
        num_utts = len(diag)
        for i in range(num_utts):
            # break
            # for i in range(num_contexts,  num_utts - num_contexts):
            utts, bio_tags, s_names, s_labels, utts_lens = [], [], [], [], []
            for j in range(max(0, i - num_contexts), min(i + num_contexts + 1, num_utts)):
                # for j in range(i - num_contexts, i + num_contexts + 1):
                utt = diag[j]['speaker'] + '：' + diag[j]['sentence']
                utts_lens.append(len(utt))
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
                    # task 2/3: predict normalized name and symptom label
                    if sn in sym2id and st in sl2id:
                        chunks_c.append([chunk[1], chunk[2], sym2id.get(sn), sl2id.get(st)])
                bio_ids_c = [bio2id.get(tag, 1) for tag in bio_tags_c]
            if len(chunks_c) > 0:
                pos += 1
            else:
                neg += 1
            begin_idx = sum(utts_lens[:min(i, num_contexts)]) + min(i, num_contexts) * len(delimiter)
            end_idx = sum(utts_lens[:min(i, num_contexts) + 1]) + (min(i, num_contexts) + 1) * len(delimiter) - 2
            instance = {
                'pid': pid,
                'sid': diag[i]['sentence_id'],
                'utts': utts_c,
                'bio_ids': bio_ids_c,
                'chunks': chunks_c,
                'bounds': [begin_idx, end_idx]
            }
            instances.append(instance)
    print('proportion of utterances containing symptom entities: {} %'.format(
        round(100 * pos / (pos + neg), 2)))
    return instances


if __name__ == '__main__':

    _num_contexts = 2

    data_dir = '../../../dataset'
    # data_dir = 'dataset'
    train_set = load_json(os.path.join(data_dir, 'V3/train_update.json'))
    dev_set = load_json(os.path.join(data_dir, 'V3/dev_update.json'))
    test_set = load_json(os.path.join(data_dir, 'V3/test_update.json'))

    # # 优先级
    # priority = {'1': 2, '0': 1, '2': 0}
    # for key, value in test_set.items():
    #     global_tmp = {}
    #     for item in value['dialogue']:
    #         local_tmp = {}
    #         for sn, st in zip(item['new_symptom_norm'], item['new_symptom_type']):
    #             if sn in local_tmp:
    #                 if priority.get(st) > priority.get(local_tmp[sn]):
    #                     local_tmp[sn] = st
    #             else:
    #                 local_tmp[sn] = st
    #         item['local_implicit_info'] = local_tmp
    #         for sn, st in local_tmp.items():
    #             if sn in global_tmp:
    #                 if priority.get(st) > priority.get(global_tmp[sn]):
    #                     global_tmp[sn] = st
    #             else:
    #                 global_tmp[sn] = st
    #     # print(global_tmp)
    #     value['global_implicit_info'] = global_tmp

    # write_json(train_set, os.path.join(data_dir, 'V3/train_update.json'), indent=4)
    # write_json(dev_set, os.path.join(data_dir, 'V3/dev_update.json'), indent=4)
    # write_json(test_set, os.path.join(data_dir, 'V3/test_update.json'), indent=4)
    #
    # for key, value in train_set.items():
    #     for item in value['dialogue']:
    #         tmp = {}
    #         for sn, st in zip(item['new_symptom_norm'], item['new_symptom_type']):
    #             if sn in tmp:
    #                 if st != tmp[sn]:
    #                     print(item['sentence'], item['new_symptom_norm'], item['new_symptom_type'])
    #             else:
    #                 tmp[sn] = st

    saved_path = 'data/c{}'.format(_num_contexts)
    os.makedirs(os.path.join(saved_path), exist_ok=True)

    mappings_path = os.path.join(data_dir, 'V3/mappings.json')

    if not os.path.exists(mappings_path):
        # 有重复，需要去重
        symptoms = list(set(pd.read_csv(os.path.join(data_dir, 'V3/symptom_norm.csv'))['norm']))
        # 症状-id 映射
        sym2id = {key: idx for idx, key in enumerate(symptoms)}
        id2sym = {idx: key for key, idx in sym2id.items()}
        # bio标签-id 映射
        bio2id = {item: idx for idx, item in enumerate(['PAD', 'O', 'B-Symptom', 'I-Symptom'])}
        id2bio = {idx: item for item, idx in bio2id.items()}
        # 症状标签-id 映射
        sl2id = {'0': 0, '1': 1, '2': 2}
        id2sl = {idx: sl for sl, idx in sl2id.items()}
        mappings = [sym2id, id2sym, bio2id, id2bio, sl2id, id2sl]
        write_json(mappings, mappings_path)
        print('writing mappings to {}'.format(mappings_path))
    else:
        sym2id, id2sym, bio2id, id2bio, sl2id, id2sl = load_json(mappings_path)

    train = make_dataset(train_set, num_contexts=_num_contexts, mode='train')
    dev = make_dataset(dev_set, num_contexts=_num_contexts, mode='dev')
    test = make_dataset(test_set, num_contexts=_num_contexts, mode='test')

    lens = np.array([len(sample['utts']) for sample in train])
    print('train/dev/test size: {}/{}/{}'.format(len(train), len(dev), len(test)))
    print('avg. utts lens: {}, 95 % utts lens: {}'.format(
        np.round(np.mean(lens), 4), np.round(np.percentile(lens, 99), 4)))

    write_json_by_line(train, os.path.join(saved_path, 'train.json'))
    write_json_by_line(dev, os.path.join(saved_path, 'dev.json'))
    write_json_by_line(test, os.path.join(saved_path, 'test.json'))
