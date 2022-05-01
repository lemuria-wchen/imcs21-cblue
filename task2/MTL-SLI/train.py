import os
from transformers import BertTokenizerFast, BertConfig, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from utils import MTLDataset, BertNER, load_json_by_line
from utils import collate_fn, get_sl_metrics, get_bio_metrics, get_sn_metrics

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--cuda_num", default='0', required=True, type=str)
args = parser.parse_args()

prefix = 'data'
model_prefix = 'saved/c2'
os.makedirs(model_prefix, exist_ok=True)

train = load_json_by_line(os.path.join(prefix, 'train.json'))
dev = load_json_by_line(os.path.join(prefix, 'dev.json'))

device = 'cuda:{}'.format(args.cuda_num) if torch.cuda.is_available() else 'cpu'

num_digits = 4

enc_dim = 768
num_bio = 12
num_sn = 331
num_sl = 3
model_name = 'bert-base-chinese'

TRAIN_BATCH_SIZE = 16
DEV_BATCH_SIZE = 16

LEARNING_RATE = 5e-5
EPOCHS = 5
NER_EPOCHS = 0
MAX_LEN = 256

ADAM_EPSILON = 1e-8
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 0
PREFIX_LEN = 4


tokenizer = BertTokenizerFast.from_pretrained(model_name)

train_set = MTLDataset(train, tokenizer, MAX_LEN)
dev_set = MTLDataset(dev, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
dev_params = {'batch_size': DEV_BATCH_SIZE, 'shuffle': False, 'num_workers': 0}

train_loader = DataLoader(train_set, collate_fn=collate_fn, **train_params)
dev_loader = DataLoader(dev_set, collate_fn=collate_fn, **dev_params)

config = BertConfig.from_pretrained(model_name, finetuning_task='ner')
# model = BertNER(config, enc_dim, num_bio, num_sn, num_sl)
# model.to(device)
model = torch.load(os.path.join(model_prefix, 'model.pkl'))

weight_decay = 0
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]

t_total = len(train_loader) * EPOCHS

optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=ADAM_EPSILON)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=t_total)

criterion_bio = torch.nn.CrossEntropyLoss(ignore_index=0)
criterion_sn = torch.nn.CrossEntropyLoss()
sl_weights = torch.tensor([4.0, 1.0, 2.0]).to(device)
criterion_sl = torch.nn.CrossEntropyLoss(weight=sl_weights)

id2bio = {idx: item for idx, item in enumerate([
    'PAD', 'O', 'B-Symptom', 'I-Symptom', 'B-Medical_Examination', 'I-Medical_Examination', 'B-Drug', 'I-Drug',
    'B-Drug_Category', 'I-Drug_Category', 'B-Operation', 'I-Operation']
)}


def train_epoch(data_loader, epoch):
    model.zero_grad()
    for step, batch in tqdm(enumerate(data_loader)):
        model.train()
        outputs = model(
            ids=batch['ids'].to(device),
            mask=batch['mask'].to(device),
            token_type_ids=batch['token_type_ids'].to(device))
        features, sn_labels, sl_labels = [], [], []
        for i, chunks in enumerate(batch['chunks']):
            for chunk in chunks:
                features.append(torch.mean(outputs[i, chunk[0]: chunk[1] + 1, :], dim=0))
                sn_labels.append(chunk[2])
                sl_labels.append(chunk[3])
        sn_labels = torch.tensor(sn_labels, dtype=torch.long, device=device)
        sl_labels = torch.tensor(sl_labels, dtype=torch.long, device=device)
        bio_outputs = model.fc_bio(outputs)
        bio_loss = criterion_bio(bio_outputs.reshape(-1, num_bio), batch['bio_ids'].to(device).reshape(-1))
        loss = 0.2 * bio_loss
        if len(features) > 0 and epoch > NER_EPOCHS:
            # symptom name
            sn_outputs = model.fc_sn(torch.stack(features))
            sn_loss = criterion_sn(sn_outputs, sn_labels)
            # symptom label
            sl_outputs = model.fc_sl(torch.stack(features))
            sl_loss = criterion_sl(sl_outputs, sl_labels)
            # add loss
            loss += 0.1 * sn_loss
            loss += 0.7 * sl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        model.zero_grad()


def dev_epoch(data_loader, epoch):
    bio_ids_true, bio_ids_pred, sn_ids_true, sn_ids_pred, sl_ids_true, sl_ids_pred = [], [], [], [], [], []
    for batch in tqdm(data_loader):
        outputs = model(
            ids=batch['ids'].to(device),
            mask=batch['mask'].to(device),
            token_type_ids=batch['token_type_ids'].to(device))
        bio_outputs = model.fc_bio(outputs)
        bio_tag_ids = torch.argmax(bio_outputs, dim=-1)
        features = []
        input_lens = (torch.sum(batch['ids'] != 0, dim=1) - PREFIX_LEN - 1).tolist()
        for i, chunks in enumerate(batch['chunks']):
            input_len = input_lens[i]
            bio_ids_true.append([id2bio.get(tag_id) for tag_id in batch['bio_ids'][i][PREFIX_LEN: input_len + PREFIX_LEN].cpu().tolist()])
            bio_ids_pred.append([id2bio.get(tag_id) for tag_id in bio_tag_ids[i][PREFIX_LEN: input_len + PREFIX_LEN].cpu().tolist()])
            for chunk in chunks:
                features.append(torch.mean(outputs[i, chunk[0]: chunk[1] + 1, :], dim=0))
                sn_ids_true.append(chunk[2])
                sl_ids_true.append(chunk[3])
        if len(features) > 0:
            sn_outputs = model.fc_sn(torch.stack(features))
            sl_outputs = model.fc_sl(torch.stack(features))
            sn_ids_pred.extend(torch.argmax(sn_outputs, dim=-1).cpu().tolist())
            sl_ids_pred.extend(torch.argmax(sl_outputs, dim=-1).cpu().tolist())
    print('Evaluation of dev epoch: {} --->'.format(epoch))
    get_bio_metrics(bio_ids_true, bio_ids_pred, num_digits)
    if epoch > NER_EPOCHS:
        get_sn_metrics(sn_ids_true, sn_ids_pred, num_digits)
        get_sl_metrics(sl_ids_true, sl_ids_pred, num_digits)


for _epoch in range(EPOCHS):
    # train
    model.train()
    train_epoch(train_loader, _epoch + 1)

    print('saving model to {}'.format(os.path.join(model_prefix, 'model_new.pkl')))
    torch.save(model, os.path.join(model_prefix, 'model_new.pkl'))

    # evaluation
    model.eval()
    with torch.no_grad():
        dev_epoch(dev_loader, _epoch + 1)
