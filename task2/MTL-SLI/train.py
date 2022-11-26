import os
from transformers import BertTokenizerFast, BertConfig
import torch
from torch.utils.data import DataLoader

from utils import MTLDataset, BertNER, load_json_by_line
from utils import load_json, collate_fn, get_sl_metrics, get_bio_metrics, get_sn_metrics


device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

# parameters
_num_contexts = 2
num_digits = 4

data_prefix = 'data/c{}'.format(_num_contexts)
mappings_path = os.path.join('../../../dataset/V3/mappings.json')

model_prefix = 'saved/c{}'.format(_num_contexts)
os.makedirs(model_prefix, exist_ok=True)

train = load_json_by_line(os.path.join(data_prefix, 'train.json'))
dev = load_json_by_line(os.path.join(data_prefix, 'dev.json'))
sym2id, id2sym, bio2id, id2bio, sl2id, id2sl = load_json(mappings_path)
print('load data from {}'.format(data_prefix))

# hyper-parameters
num_bio = len(bio2id)
num_sn = len(sym2id)
num_sl = len(sl2id)

enc_dim = 768
model_name = 'bert-base-chinese'

TRAIN_BATCH_SIZE = 32
DEV_BATCH_SIZE = 32

LEARNING_RATE = 1e-5
EPOCHS = 20
NER_EPOCHS = 5
MAX_LEN = 256

MAX_GRAD_NORM = 1.0

# load tokenizer, dataset and model
tokenizer = BertTokenizerFast.from_pretrained(model_name)

train_set = MTLDataset(train, tokenizer, MAX_LEN)
dev_set = MTLDataset(dev, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
dev_params = {'batch_size': DEV_BATCH_SIZE, 'shuffle': False, 'num_workers': 0}

train_loader = DataLoader(train_set, collate_fn=collate_fn, **train_params)
dev_loader = DataLoader(dev_set, collate_fn=collate_fn, **dev_params)

config = BertConfig.from_pretrained(model_name, finetuning_task='ner')
model = BertNER(config, enc_dim, num_bio, num_sn, num_sl)
model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

criterion_bio = torch.nn.CrossEntropyLoss(ignore_index=0)
criterion_sn = torch.nn.CrossEntropyLoss()
criterion_sl = torch.nn.CrossEntropyLoss()


def get_entity_bio_test(bio_list):
    chunks = []
    i = 0
    n = len(bio_list)
    while i < n:
        while i < n and (bio_list[i] != 2):
            i += 1
        start_idx = i
        while i < n and (bio_list[i] != 1):
            i += 1
        end_idx = i
        if start_idx < n and end_idx < n + 1:
            chunks.append((start_idx, end_idx - 1))
    return chunks


def train_epoch(data_loader, epoch):
    for step, batch in enumerate(data_loader):
        # break
        # print(tokenizer.convert_ids_to_tokens(batch['ids'][62]))
        # print(batch['bio_ids'][62])
        # print(batch['chunks'][62])
        # print(id2sym.get('31'))
        # get_entity_bio_test(batch['bio_ids'][62].tolist())
        outputs = model(
            ids=batch['ids'].to(device),
            mask=batch['mask'].to(device),
            token_type_ids=batch['token_type_ids'].to(device))
        # bsz, seq_len, dim
        # bsz, seq_len,
        bio_outputs = model.bio_forward(outputs)
        bio_loss = criterion_bio(bio_outputs.reshape(-1, num_bio), batch['bio_ids'].to(device).reshape(-1))
        loss = bio_loss
        if epoch > NER_EPOCHS:
            features, sn_labels, sl_labels = [], [], []
            for i, chunks in enumerate(batch['chunks']):
                for chunk in chunks:
                    features.append(torch.mean(outputs[i, chunk[0]: chunk[1] + 1, :], dim=0))
                    sn_labels.append(chunk[2])
                    sl_labels.append(chunk[3])
            sn_labels = torch.tensor(sn_labels, dtype=torch.long, device=device)
            sl_labels = torch.tensor(sl_labels, dtype=torch.long, device=device)
            if len(features) > 0:
                # stack the features
                stacked_features = torch.stack(features)
                # symptom name
                sn_outputs = model.sn_forward(stacked_features)
                sn_loss = criterion_sn(sn_outputs, sn_labels)
                # symptom label
                sl_outputs = model.sl_forward(stacked_features)
                sl_loss = criterion_sl(sl_outputs, sl_labels)
                # sum the loss
                loss += sn_loss + sl_loss
                if step % 50 == 0:
                    print('step:{}, bio loss: {}, sn loss: {}, sl loss: {}'.format(
                        step,
                        round(bio_loss.item(), num_digits),
                        round(sn_loss.item(), num_digits),
                        round(sl_loss.item(), num_digits),
                    ))
        else:
            if step % 50 == 0:
                print('step:{}, bio loss: {}'.format(step, round(bio_loss.item(), num_digits)))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()


def dev_epoch(data_loader, epoch):
    bio_ids_true, bio_ids_pred, sn_ids_true, sn_ids_pred, sl_ids_true, sl_ids_pred = [], [], [], [], [], []
    for batch in data_loader:
        outputs = model(
            ids=batch['ids'].to(device),
            mask=batch['mask'].to(device),
            token_type_ids=batch['token_type_ids'].to(device))
        bio_outputs = model.fc_bio(outputs)
        bio_tag_ids = torch.argmax(bio_outputs, dim=-1)
        features = []
        input_lens = (torch.sum(batch['ids'] != 0, dim=1)).tolist()
        for i, chunks in enumerate(batch['chunks']):
            input_len = input_lens[i]
            # 这个时候计算的指标不完全准确，因为此时计算的是 tokenized 后的指标，但是已经可以作为参考
            bio_ids_true.extend([
                id2bio.get(str(tag_id)) for tag_id in batch['bio_ids'][i][1: input_len - 1].cpu().tolist()])
            bio_ids_pred.extend([
                id2bio.get(str(tag_id)) for tag_id in bio_tag_ids[i][1: input_len - 1].cpu().tolist()])
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
    m = get_bio_metrics(bio_ids_true, bio_ids_pred, num_digits)
    if epoch > NER_EPOCHS:
        m += get_sn_metrics(sn_ids_true, sn_ids_pred, num_digits)
        m += get_sl_metrics(sl_ids_true, sl_ids_pred, num_digits)
    return m


best_metric = 0
best_epoch = 1

for _epoch in range(1, EPOCHS + 1):
    # train
    model.train()
    train_epoch(train_loader, _epoch)

    print('saving model to {}'.format(os.path.join(model_prefix, 'model_{}.pkl'.format(_epoch))))
    torch.save(model, os.path.join(model_prefix, 'model_{}.pkl'.format(_epoch)))

    # evaluation
    model.eval()
    with torch.no_grad():
        metric = dev_epoch(dev_loader, _epoch)
        if metric > best_metric:
            best_metric = metric
            best_epoch = _epoch

print('best epoch: {}, best metric: {}'.format(best_epoch, best_metric))
