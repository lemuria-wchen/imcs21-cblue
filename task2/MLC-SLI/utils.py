import torch
from torch.utils.data import Dataset
import transformers
import json


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        inputs = self.tokenizer.encode_plus(
            x,
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
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(y, dtype=torch.float)}


class BERTClass(torch.nn.Module):
    def __init__(self, model_name: str, num_labels: int, target: str):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(model_name)
        self.l2 = torch.nn.Dropout(p=0.3)
        if target == 'exp':
            self.l3 = torch.nn.Linear(768, num_labels)
        else:
            self.l3 = torch.nn.Linear(768, num_labels * 3)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json(data, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
