import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm
from collections import defaultdict

from utils import MTLDataset, collate_fn_test, load_json_by_line, write_json, get_entity_bio

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='test', required=True, type=str)
parser.add_argument("--cuda_num", default='0', required=True, type=str)
args = parser.parse_args()


prefix = 'data'
model_prefix = 'saved'

# load test set
test = load_json_by_line(os.path.join(prefix, '{}.json'.format(args.dataset)))

# load symptoms
id2sym = {key: value for key, value in pd.read_csv('../../../dataset/symptom_norm.csv')['norm'].items()}
id2bio = {idx: item for idx, item in enumerate([
    'PAD', 'O', 'B-Symptom', 'I-Symptom', 'B-Medical_Examination', 'I-Medical_Examination', 'B-Drug',
    'I-Drug', 'B-Drug_Category', 'I-Drug_Category', 'B-Operation', 'I-Operation'])}
id2sl = {0: '0', 1: '1', 2: '2'}

# settings
device = 'cuda:{}'.format(args.cuda_num) if torch.cuda.is_available() else 'cpu'

model_name = 'bert-base-chinese'

TEST_BATCH_SIZE = 32
MAX_LEN = 128
PREFIX_LEN = 4
# data loader
tokenizer = BertTokenizerFast.from_pretrained(model_name)
test_set = MTLDataset(test, tokenizer, MAX_LEN, mode='test')

test_params = {'batch_size': TEST_BATCH_SIZE, 'shuffle': False, 'num_workers': 0}
test_loader = DataLoader(test_set, collate_fn=collate_fn_test, **test_params)

# load model
model = torch.load(os.path.join(model_prefix, 'model.pkl'))
model.to(device)

# inference
preds = []
model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader):
        outputs = model(
            ids=batch['ids'].to(device),
            mask=batch['mask'].to(device),
            token_type_ids=batch['token_type_ids'].to(device))
        bio_outputs = model.fc_bio(outputs)
        bio_tag_ids = torch.argmax(bio_outputs, dim=-1)
        input_lens = (torch.sum(batch['ids'] != 0, dim=1) - PREFIX_LEN - 1).tolist()
        for i in range(len(input_lens)):
            input_len = input_lens[i]
            bio_tag_id = bio_tag_ids[i]
            bio_tags = [id2bio.get(tag_id) for tag_id in bio_tag_id[PREFIX_LEN: input_len + PREFIX_LEN].cpu().tolist()]
            chunks = get_entity_bio(bio_tags)
            features = []
            for chunk in chunks:
                if chunk[0] == 'Symptom':
                    features.append(torch.mean(outputs[i, PREFIX_LEN + chunk[1]: PREFIX_LEN + chunk[2] + 1, :], dim=0))
            if len(features) > 0:
                sn_outputs = model.fc_sn(torch.stack(features))
                sl_outputs = model.fc_sl(torch.stack(features))
                sn_ids = torch.argmax(sn_outputs, dim=-1).cpu().tolist()
                sl_ids = torch.argmax(sl_outputs, dim=-1).cpu().tolist()
                pred = {id2sym.get(sn_id): id2sl.get(sl_id) for sn_id, sl_id in zip(sn_ids, sl_ids)}
            else:
                pred = {}
            preds.append(pred)

pids = [sample['pid'] for sample in test]

final = defaultdict(dict)
for pid, pred in zip(pids, preds):
    final[pid].update(pred)

write_json(final, path='{}_imp_pred.json'.format(args.dataset), indent=4)
