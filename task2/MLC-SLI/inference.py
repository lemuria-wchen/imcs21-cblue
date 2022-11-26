import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from utils import CustomDataset, load_json, load_json_by_line, write_json, collate_fn

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='test', required=False, type=str)
parser.add_argument("--target", default='exp', required=False, type=str)
args = parser.parse_args()


# define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# define path
prefix = 'data'
model_prefix = './saved'
data_dir = '../../../dataset/V3'

# load test set (or dev set)
test = load_json_by_line(os.path.join(prefix, args.target, '{}.json'.format(args.dataset)))

# load normalized symptom
mappings_path = os.path.join(data_dir, 'mappings.json')
sym2id, id2sym, _, _, sl2id, id2sl = load_json(mappings_path)
num_labels = len(sym2id) if args.target == 'exp' else len(sym2id) * len(sl2id)


# load model
model_name = 'bert-base-chinese'
# model_name = 'hfl/chinese-bert-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(model_name)

# load model
model = torch.load(os.path.join(model_prefix, args.target, 'model.pkl'))
model.to(device)


# load test set
MAX_LEN = 128
test_set = CustomDataset(test, tokenizer, MAX_LEN, num_labels)

# load test dataloader
TEST_BATCH_SIZE = 16

test_params = {
    'batch_size': TEST_BATCH_SIZE,
    'shuffle': False,
    'num_workers': 1
}

test_loader = DataLoader(test_set, collate_fn=collate_fn, **test_params)

# inference
priority = {'1': 2, '0': 1, '2': 0}
pred_results = {}

if args.target == 'exp':
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            outputs = model(
                ids=batch['ids'].to(device),
                mask=batch['mask'].to(device),
                token_type_ids=batch['token_type_ids'].to(device))
            outputs = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5
            for i in range(len(outputs)):
                pid = batch['pids'][i]
                # print(tokenizer.convert_ids_to_tokens(batch['ids'][i]))
                predicted_symptoms,  = np.where(outputs[i])
                sns = []
                for predicted_symptom in predicted_symptoms:
                    sns.append(id2sym.get(str(predicted_symptom)))
                pred_results[pid] = {sns}
else:
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # break
            outputs = model(
                ids=batch['ids'].to(device),
                mask=batch['mask'].to(device),
                token_type_ids=batch['token_type_ids'].to(device))
            outputs = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5
            for i in range(len(outputs)):
                pid = batch['pids'][i]
                sid = batch['sids'][i]
                if pid not in pred_results:
                    pred_results[pid] = {}
                if sid not in pred_results.get(pid):
                    pred_results[pid][sid] = {}
                # print(tokenizer.convert_ids_to_tokens(batch['ids'][i]))
                predicted_symptoms,  = np.where(outputs[i])
                for predicted_symptom in predicted_symptoms:
                    sn = id2sym.get(str(predicted_symptom // len(sl2id)))
                    sl = id2sl.get(str(predicted_symptom % len(sl2id)))
                    if sn in pred_results[pid][sid]:
                        if priority.get(sl) > priority.get(pred_results[pid][sid][sn]):
                            pred_results[pid][sid][sn] = sl
                    else:
                        pred_results[pid][sid][sn] = sl

    # update global symptoms
    for pid, d in pred_results.items():
        tmp = {}
        for sid, value in d.items():
            for sn, sl in value.items():
                if sn in tmp:
                    if priority.get(sl) > priority.get(tmp[sn]):
                        tmp[sn] = sl
                else:
                    tmp[sn] = sl
        d['global'] = tmp


write_json(pred_results, path='pred_results_{}.json'.format(args.target), indent=4)
