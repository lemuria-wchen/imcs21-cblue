import json
import os


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


data_dir = '../../../dataset'
# data_dir = 'dataset'
train_set = load_json(os.path.join(data_dir, 'train.json'))
dev_set = load_json(os.path.join(data_dir, 'dev.json'))
test_set = load_json(os.path.join(data_dir, 'test.json'))

saved_path = 'THUCNews/data'
os.makedirs(saved_path, exist_ok=True)

tags = [
    'Request-Etiology', 'Request-Precautions', 'Request-Medical_Advice', 'Inform-Etiology', 'Diagnose',
    'Request-Basic_Information', 'Request-Drug_Recommendation', 'Inform-Medical_Advice',
    'Request-Existing_Examination_and_Treatment', 'Inform-Basic_Information', 'Inform-Precautions',
    'Inform-Existing_Examination_and_Treatment', 'Inform-Drug_Recommendation', 'Request-Symptom',
    'Inform-Symptom', 'Other'
]
tag2id = {tag: idx for idx, tag in enumerate(tags)}


def make_tag(path):
    with open(path, 'w', encoding='utf-8') as f:
        for tag in tags:
            f.write(tag + '\n')


def make_data(samples, path):
    out = ''
    for pid, sample in samples.items():
        for sent in sample['dialogue']:
            x = sent['speaker'] + '：' + sent['sentence']
            assert sent['dialogue_act'] in tag2id
            y = tag2id.get(sent['dialogue_act'])
            out += x + '\t' + str(y) + '\n'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(out)
    return out


make_tag(os.path.join(saved_path, 'class.txt'))

make_data(train_set, os.path.join(saved_path, 'train.txt'))
make_data(dev_set, os.path.join(saved_path, 'dev.txt'))
make_data(test_set, os.path.join(saved_path, 'test.txt'))
