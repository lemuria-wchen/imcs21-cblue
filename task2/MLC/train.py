import os
import torch
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

from utils import CustomDataset, BERTClass, load_json


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(12345)

# load train/dev set
prefix = './data'
model_prefix = './saved'

os.makedirs(model_prefix, exist_ok=True)

train = load_json(os.path.join(prefix, 'processed', 'train_set.json'))
dev = load_json(os.path.join(prefix, 'processed', 'dev_set.json'))

# Defining some key variables that will be used later on in the training
MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

train_set = CustomDataset(train, tokenizer, MAX_LEN)
dev_set = CustomDataset(dev, tokenizer, MAX_LEN)

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

weights = [sample[3] for sample in train]
print('weights: max = {}, min = {}, mean = {}'.format(np.max(weights), np.min(weights), np.mean(weights)))
sampler = WeightedRandomSampler(weights, num_samples=len(train), replacement=True)

train_loader = DataLoader(train_set, sampler=sampler, **train_params)
dev_loader = DataLoader(dev_set, **dev_params)

model = BERTClass()
model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def train_epoch(_epoch):
    model.train()
    for _, data in tqdm(enumerate(train_loader)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)
        outputs = model(ids, mask, token_type_ids)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _ % 100 == 0:
            print(f'Epoch: {_epoch + 1}, Loss:  {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model, os.path.join(model_prefix, 'model_{}.pkl'.format(_epoch + 1)))


def validate():
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(dev_loader), total=len(dev_set) // VALID_BATCH_SIZE + 1):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    fin_outputs = np.array(fin_outputs) >= 0.5
    accuracy = metrics.accuracy_score(fin_targets, fin_outputs)
    f1_score_micro = metrics.f1_score(fin_targets, fin_outputs, average='micro')
    f1_score_macro = metrics.f1_score(fin_targets, fin_outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")


for epoch in range(EPOCHS):
    train_epoch(epoch)
    validate()