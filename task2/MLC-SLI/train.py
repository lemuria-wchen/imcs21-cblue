import os
import torch
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.utils.data.sampler import WeightedRandomSampler
from utils import CustomDataset, BERTClass, load_json, load_json_by_line, collate_fn
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--target", default='imp', required=True, type=str)
args = parser.parse_args()

# args.target = 'exp'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load train/dev set
prefix = 'data'
train = load_json_by_line(os.path.join(prefix, args.target, 'train.json'))
dev = load_json_by_line(os.path.join(prefix, args.target, 'dev.json'))

data_dir = '../../../dataset'
mappings_path = os.path.join(data_dir, 'V3/mappings.json')
sym2id, _, _, _, sl2id, _ = load_json(mappings_path)
num_labels = len(sym2id) if args.target == 'exp' else len(sym2id) * len(sl2id)

model_prefix = os.path.join('saved', args.target)
os.makedirs(model_prefix, exist_ok=True)

# Defining some key variables that will be used later on in the training
model_name = 'bert-base-chinese'
# model_name = 'hfl/chinese-bert-wwm-ext'
MAX_LEN = 256 if args.target == 'exp' else 128
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
EPOCHS = 100 if args.target == 'exp' else 50
LEARNING_RATE = 1e-5

tokenizer = BertTokenizer.from_pretrained(model_name)

train_set = CustomDataset(train, tokenizer, MAX_LEN, num_labels)
dev_set = CustomDataset(dev, tokenizer, MAX_LEN, num_labels)

train_params = {
    'batch_size': TRAIN_BATCH_SIZE,
    # 'shuffle': True,
    'num_workers': 1
}

dev_params = {
    'batch_size': VALID_BATCH_SIZE,
    'shuffle': False,
    'num_workers': 1
}

weights = [sample['weight'] for sample in train]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

if args.target == 'exp':
    train_loader = DataLoader(train_set, collate_fn=collate_fn, **train_params)
else:
    train_loader = DataLoader(train_set, sampler=sampler, collate_fn=collate_fn, **train_params)
dev_loader = DataLoader(dev_set, collate_fn=collate_fn, **dev_params)

model = BERTClass(model_name, num_labels)
model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def train_epoch(_epoch):
    for step, data in enumerate(train_loader):
        # break
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)
        outputs = model(ids, mask, token_type_ids)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        # if step % 10 == 0:
        #     print('step: {}, loss: {}'.format(
        #         step, loss.item()
        #     ))
        optimizer.step()


def validate(_epoch):
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(dev_loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    fin_outputs = np.array(fin_outputs) >= 0.5
    accuracy = metrics.accuracy_score(fin_targets, fin_outputs)
    f1_score_micro = metrics.f1_score(fin_targets, fin_outputs, average='micro', zero_division=0)
    f1_score_macro = metrics.f1_score(fin_targets, fin_outputs, average='macro', zero_division=0)
    print("Dev epoch: {}, Acc: {}, Micro F1: {}, Macro F1: {}".format(
        _epoch + 1, round(accuracy, 4),  round(f1_score_micro, 4), round(f1_score_macro, 4)))
    return accuracy, f1_score_micro, f1_score_macro


print('total steps: {}'.format(len(train_loader) * EPOCHS))
best_micro_f1 = -1

for epoch in range(EPOCHS):
    # break
    model.train()
    train_epoch(epoch)
    model.eval()
    with torch.no_grad():
        _, micro_f1, _ = validate(epoch)
    if micro_f1 > best_micro_f1:
        print('saving model to : {}'.format(model_prefix))
        torch.save(model, os.path.join(model_prefix, 'model.pkl'))
        best_micro_f1 = micro_f1
