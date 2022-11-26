import torch
from torch.utils.data import Dataset
import transformers
import json
from torch.nn.utils.rnn import pad_sequence


def collate_fn(samples):
    ids = pad_sequence([sample['ids'] for sample in samples], padding_value=0, batch_first=True)
    mask = pad_sequence([sample['mask'] for sample in samples], padding_value=0, batch_first=True)
    token_type_ids = pad_sequence([sample['token_type_ids'] for sample in samples], padding_value=0, batch_first=True)
    targets = torch.stack([sample['targets'] for sample in samples])
    sids = [sample['sid'] for sample in samples]
    pids = [sample['pid'] for sample in samples]
    return {
        'sids': sids,
        'pids': pids,
        'ids': ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'targets': targets,
    }


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, num_labels):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_labels = num_labels

    def __len__(self):
        return len(self.data)

    def make_label(self, label):
        y = [0] * self.num_labels
        for _label in label:
            y[_label] = 1
        return y

    def __getitem__(self, index):
        pid = self.data[index]['pid']
        if 'sid' in self.data[index]:
            sid = self.data[index]['sid']
        else:
            sid = None
        utts = self.data[index]['utts']
        label = self.make_label(self.data[index]['label'])
        inputs = self.tokenizer.encode_plus(
            utts,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return {
            'pid': pid,
            'sid': sid,
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(label, dtype=torch.float)}


class BERTClass(torch.nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.3):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(model_name)
        self.l2 = torch.nn.Dropout(p=dropout)
        self.l3 = torch.nn.Linear(self.l1.embeddings.word_embeddings.embedding_dim, num_labels)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json_v1(data, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


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
