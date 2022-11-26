import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm
# from collections import defaultdict

from utils import load_json, MTLDataset, collate_fn_test, load_json_by_line, write_json, get_entity_bio


device = 'cuda' if torch.cuda.is_available() else 'cpu'

_num_contexts = 1

data_prefix = 'data/c{}'.format(_num_contexts)
model_prefix = 'saved/c{}'.format(_num_contexts)

# load test set
test = load_json_by_line(os.path.join(data_prefix, 'test.json'))

mappings_path = os.path.join('../../../dataset/V3/mappings.json')
sym2id, id2sym, bio2id, id2bio, sl2id, id2sl = load_json(mappings_path)
print('load data from {}'.format(data_prefix))


# load symptoms
model_name = 'bert-base-chinese'

TEST_BATCH_SIZE = 64
MAX_LEN = 128

# data loader
tokenizer = BertTokenizerFast.from_pretrained(model_name)
test_set = MTLDataset(test, tokenizer, MAX_LEN, mode='test')

test_params = {'batch_size': TEST_BATCH_SIZE, 'shuffle': False, 'num_workers': 0}
test_loader = DataLoader(test_set, collate_fn=collate_fn_test, **test_params)

# load model
# c0 -> best epoch: 12
# c1 -> best epoch: 16
# c2 -> best epoch: 12
model_path = os.path.join(model_prefix, 'model_best.pkl')
model = torch.load(model_path)
model.to(device)

# inference
priority = {'1': 2, '0': 1, '2': 0}
pred_results = {}

model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader):
        # break
        outputs = model(
            ids=batch['ids'].to(device),
            mask=batch['mask'].to(device),
            token_type_ids=batch['token_type_ids'].to(device))
        bio_outputs = model.fc_bio(outputs)
        bio_tag_ids = torch.argmax(bio_outputs, dim=-1)
        for i in range(len(bio_tag_ids)):
            pid = batch['pids'][i]
            sid = batch['sids'][i]
            if pid not in pred_results:
                pred_results[pid] = {}
            if sid not in pred_results.get(pid):
                pred_results[pid][sid] = {}
            ids = batch['ids'][i]
            begin_idx, end_idx = batch['bounds'][i]
            # print(tokenizer.convert_ids_to_tokens(batch['ids'][i])[begin_idx: end_idx + 1])
            # char2id = batch['char2id'][i]
            input_len = torch.sum(ids != 0).item()
            # 预测 BIO 标签
            bio_tag_id = bio_tag_ids[i]
            bio_tags = [id2bio.get(str(tag_id)) for tag_id in bio_tag_id[: input_len].cpu().tolist()]
            # bio_tags = [id2bio.get(str(tag_id)) for tag_id in bio_tag_id[begin_idx: end_idx].cpu().tolist()]
            # 转化为原始的 BIO 标签（非必需）
            # decoded_bio_tags = []
            # for idx in range(len(char2id)):
            #     decoded_bio_tags.append(bio_tags[char2id.get(idx)])
            # decoded_bio_tags.extend(['O'] * (len(test[j]['utts']) - len(decoded_bio_tags)))
            # assert len(decoded_bio_tags) == len(test[j]['utts'])
            # decoded_bio_tags_list.append(decoded_bio_tags)
            # 预测症状标准化名称和标签
            chunks = get_entity_bio(bio_tags)
            features = []
            for chunk in chunks:
                if chunk[0] == 'Symptom':
                    if chunk[1] >= begin_idx and chunk[2] <= end_idx:
                        features.append(torch.mean(outputs[i, chunk[1]: chunk[2] + 1, :], dim=0))
            if len(features) > 0:
                sn_outputs = model.fc_sn(torch.stack(features))
                sl_outputs = model.fc_sl(torch.stack(features))
                sn_ids = torch.argmax(sn_outputs, dim=-1).cpu().tolist()
                sl_ids = torch.argmax(sl_outputs, dim=-1).cpu().tolist()
                # sym_norm_type = [{id2sym.get(str(sn_id)): id2sl.get(str(sl_id))} for sn_id, sl_id in zip(sn_ids, sl_ids)]
                for sn_id, sl_id in zip(sn_ids, sl_ids):
                    sn = id2sym.get(str(sn_id))
                    sl = id2sl.get(str(sl_id))
                    if sn in pred_results[pid][sid]:
                        if priority.get(sl) > priority.get(pred_results[pid][sid][sn]):
                            pred_results[pid][sid][sn] = sl
                    else:
                        pred_results[pid][sid][sn] = sl
            else:
                pred_results[pid][sid] = {}

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

write_json(pred_results, path='pred_results_c{}.json'.format(_num_contexts), indent=4)
