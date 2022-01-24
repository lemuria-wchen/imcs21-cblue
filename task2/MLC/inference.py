import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from utils import CustomDataset, BERTClass, load_json, write_json


prefix = './data'
prefix2 = '../../dataset'
model_prefix = './saved'

# define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load normalized symptom
id2sym = {key: value for key, value in pd.read_csv(os.path.join(prefix2, 'symptom_norm.csv'))['norm'].items()}

# load model
best_epoch = 12
model = torch.load(os.path.join(model_prefix, 'model_{}.pkl'.format(best_epoch)))

# load test set
test = load_json(os.path.join(prefix, 'processed', 'test_set.json'))

MAX_LEN = 256
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

test_set = CustomDataset(test, tokenizer, MAX_LEN)

TEST_BATCH_SIZE = 64

test_params = {
    'batch_size': TEST_BATCH_SIZE,
    'shuffle': False,
    'num_workers': 1
}

test_loader = DataLoader(test_set, **test_params)


def inference():
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(test_loader), total=len(test_set) // TEST_BATCH_SIZE + 1):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            _targets = data['targets'].to(device, dtype=torch.float)
            _outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(_targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(_outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


outputs, targets = inference()
outputs = np.array(outputs) >= 0.5

# convert to true label
sids = []
for i in range(len(test)):
    if test[i][2] not in sids:
        sids.append(test[i][2])

final = {}
start = 0
end = 0

# evaluate
for i in range(len(sids)):
    sid = sids[i]
    while end < len(test) and test[end][2] == sid:
        end += 1
    _, labels = np.where(outputs[start: end])
    pl = {}
    for label in labels:
        pl[id2sym[label // 3]] = str(int(label % 3))
    final[str(sid)] = pl
    start = end


write_json(final, os.path.join(prefix, 'submission_track1_task2.json'))

