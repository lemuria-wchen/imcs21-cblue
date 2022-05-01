import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from seqeval import metrics as seq_m
from sklearn import metrics as skl_m
from transformers.modeling_bert import BertModel


def collate_fn(samples):
    ids = pad_sequence([sample['ids'] for sample in samples], padding_value=0, batch_first=True)
    mask = pad_sequence([sample['mask'] for sample in samples], padding_value=0, batch_first=True)
    token_type_ids = pad_sequence([sample['token_type_ids'] for sample in samples], padding_value=0, batch_first=True)
    bio_ids = pad_sequence([sample['bio_ids'] for sample in samples], padding_value=0, batch_first=True)
    chunks = [sample['chunks'] for sample in samples]
    return {
        'ids': ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'bio_ids': bio_ids,
        'chunks': chunks,
    }


def collate_fn_test(samples):
    ids = pad_sequence([sample['ids'] for sample in samples], padding_value=0, batch_first=True)
    mask = pad_sequence([sample['mask'] for sample in samples], padding_value=0, batch_first=True)
    token_type_ids = pad_sequence([sample['token_type_ids'] for sample in samples], padding_value=0, batch_first=True)
    return {
        'ids': ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
    }


class MTLDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, mode='train'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utts = self.data[index]['utts']
        # tokenize
        inputs = self.tokenizer.encode_plus(
            utts,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_offsets_mapping=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        offset_mapping = inputs['offset_mapping']

        bio_ids = self.data[index]['bio_ids']
        char2id = {}
        mapped_bio_ids = []
        chunks = self.data[index]['chunks']
        mapped_chunks = []

        if self.mode != 'test':
            for idx, om in enumerate(offset_mapping):
                if om is None:
                    mapped_bio_ids.append(0)
                else:
                    for offset in range(om[0], om[1]):
                        char2id[offset] = idx
                    mapped_bio_ids.append(bio_ids[om[0]])
            assert len(ids) == len(mapped_bio_ids)

            for chunk in chunks:
                if chunk[0] in char2id and chunk[1] in char2id:
                    mapped_chunks.append([char2id.get(chunk[0]), char2id.get(chunk[1]), chunk[2], chunk[3]])

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'bio_ids': torch.tensor(mapped_bio_ids, dtype=torch.long),
            'chunks': mapped_chunks,
        }


class BertNER(torch.nn.Module):
    def __init__(self, config, enc_dim: int, num_bio: int,
                 num_sn: int, num_sl: int):
        super(BertNER, self).__init__()
        self.encoder = BertModel(config=config)
        self.fc_bio = torch.nn.Linear(enc_dim, num_bio)
        self.fc_sn = torch.nn.Linear(enc_dim, num_sn)
        self.fc_sl = torch.nn.Linear(enc_dim, num_sl)

    def forward(self, ids, mask, token_type_ids):
        outputs = self.encoder(ids, attention_mask=mask, token_type_ids=token_type_ids)
        return outputs[0]


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json(data, path: str, indent=None):
    with open(path, 'w', encoding='utf-8') as f:
        if not indent:
            json.dump(data, f, ensure_ascii=False)
        else:
            json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json_by_line(path: str):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def write_json_by_line(data, path: str):
    content = ''
    for d in data:
        content += json.dumps(d, ensure_ascii=False) + '\n'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def get_entity_bio(seq):
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_bio_metrics(y_true, y_pred, digits=4):
    p = seq_m.precision_score(y_true, y_pred)
    r = seq_m.recall_score(y_true, y_pred)
    f1 = seq_m.f1_score(y_true, y_pred)
    print('ner metric p/r/f1: {}/{}/{}'.format(round(p, digits), round(r, digits), round(f1, digits)))


def get_sn_metrics(y_true, y_pred, digits=4):
    p = skl_m.precision_score(y_true, y_pred, average='macro', zero_division=0)
    r = skl_m.recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = skl_m.f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc = skl_m.accuracy_score(y_true, y_pred)
    print('sn metric p/r/f1/acc: {}/{}/{}/{}'.format(
        round(p, digits), round(r, digits), round(f1, digits), round(acc, digits)))


def get_sl_metrics(y_true, y_pred, digits=4):
    f1 = skl_m.f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc = skl_m.accuracy_score(y_true, y_pred)
    t_f1 = skl_m.classification_report(y_true, y_pred, output_dict=True)
    print('sl metric pos/neg/ns/f1/acc: {}/{}/{}/{}/{}'.format(
        round(t_f1['1']['f1-score'], digits),
        round(t_f1['0']['f1-score'], digits),
        round(t_f1['2']['f1-score'], digits),
        round(f1, digits),
        round(acc, digits)))
