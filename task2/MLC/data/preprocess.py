import os
import json
import pandas as pd
from collections import defaultdict

def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json(data, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        if isinstance(data, list):
            print('writing {} records to {}'.format(len(data), path))


prefix = '../../../dataset'

# load train/test.json
train = load_json(os.path.join(prefix, 'train.json'))
test = load_json(os.path.join(prefix, 'test.json'))


# load split.csv
split = defaultdict(list)
for _, row in pd.read_csv(os.path.join(prefix, 'split.csv'))[['example_id', 'split']].iterrows():
    split[row['split']].append(row['example_id'])

# load normalized symptom
sym2id = {value: key for key, value in pd.read_csv(os.path.join(prefix, 'symptom_norm.csv'))['norm'].items()}
num_labels = len(sym2id) * 3


# make label
def make_label(symptom_norm, symptom_type):
    assert len(symptom_norm) == len(symptom_type)
    label = [0] * num_labels
    for i in range(len(symptom_norm)):
        if sym2id.get(symptom_norm[i]) is not None:
            label[sym2id.get(symptom_norm[i]) * 3 + int(symptom_type[i])] = 1
    return label


# make train/dev/test set, extract input & output
# note: one can use all the information in the train set to build more complicated models
def make_dataset(sample):
    out = []
    for i in range(len(sample)):
        _sample, sid = sample[i]
        for sent in _sample['dialogue']:
            x = sent['speaker'] + ':' + sent['sentence']
            if 'symptom_norm' in sent and 'symptom_type' in sent:
                y = make_label(sent['symptom_norm'], sent['symptom_type'])
                # let the sample have a greater probability to be sampled, the constant can be a more complex function
                if len(sent['symptom_norm']) == 0 and len(sent['symptom_type']) == 0:
                    weight = 1
                else:
                    weight = 20
            else:
                y = []
                weight = 1
            out.append((x, y, sid, weight))
    return out


# make dataset for train/dev/test set, note that in test set, the label is empty
train_set = make_dataset([(train[str(sid)], sid) for sid in split['train']])
dev_set = make_dataset([(train[str(sid)], sid) for sid in split['dev']])
test_set = make_dataset([(test[str(sid)], sid) for sid in split['test']])

os.makedirs(os.path.join('processed'), exist_ok=True)

write_json(train_set, os.path.join('processed', 'train_set.json'))
write_json(dev_set, os.path.join('processed', 'dev_set.json'))
write_json(test_set, os.path.join('processed', 'test_set.json'))

print('train/dev/test size: {}/{}/{}'.format(len(split['train']), len(split['dev']), len(split['test'])))