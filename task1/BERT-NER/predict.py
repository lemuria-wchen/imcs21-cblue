import os
import logging
import argparse
from tqdm import tqdm
import json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from utils import init_logger, load_tokenizer, get_seq_labels, MODEL_CLASSES

logger = logging.getLogger(__name__)


def get_device(pred_config):
    """获得设备类型"""
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(pred_config):
    """获取训练参数"""
    return torch.load(os.path.join(pred_config.model_dir, 'training_args.bin'))


def load_model(pred_config, args, device):
    """加载模型"""
    print('==================================模型是:',MODEL_CLASSES[args.model_type][1])
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = MODEL_CLASSES[args.model_type][1].from_pretrained(args.model_dir,
                                                                  args=args,
                                                                  seq_label_lst=get_seq_labels(args))
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model


def read_input_file(input_path):
    """读取预测的文件"""
    lines = []
    eids = []
    sids = []
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for k, v in data.items():
        for sent in data[k]['dialogue']:
            words = list(sent['speaker'] + '：' + sent['sentence'])
            lines.append(words)
            eids.append(k)
            sids.append(sent['sentence_id'])

    return (lines, eids, sids)


def convert_input_file_to_tensor_dataset(lines,
                                         tokenizer,
                                         pad_token_label_id,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    """将原始数据转换为特征"""

    # 设置最大长度
    max_seq_len = len(max(lines, key=len)) + 2

    # 设置一些符号
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_seq_label_mask = []

    for words in lines:
        tokens = []
        seq_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # 处理UNK
            tokens.extend(word_tokens)
            seq_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

        # 添加 [SEP]
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        seq_label_mask += [pad_token_label_id]

        # 添加 [CLS]
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        seq_label_mask = [pad_token_label_id] + seq_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # 真实token：1；pad token：0
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # pad
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        seq_label_mask = seq_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_seq_label_mask.append(seq_label_mask)

    # 转换为Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_seq_label_mask = torch.tensor(all_seq_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_seq_label_mask)

    return dataset


def predict(pred_config):
    """预测"""
    # 读取模型和参数
    args = get_args(pred_config)
    device = get_device(pred_config)
    model = load_model(pred_config, args, device)
    logger.info(args)

    seq_label_lst = get_seq_labels(args)

    pad_token_label_id = args.ignore_index
    tokenizer = load_tokenizer(args)
    # 读取文本序列列表，example_ids, sentence_ids
    (lines, eids, sids) = read_input_file(
            os.path.join(pred_config.test_input_file))
    dataset = convert_input_file_to_tensor_dataset(lines, tokenizer, pad_token_label_id)

    # 预测
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    all_seq_label_mask = None
    seq_preds = None
    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2],
                      "seq_labels_ids": None}
            outputs = model(**inputs)
            _, seq_logits = outputs[:2]

            if seq_preds is None:
                if args.use_crf:
                    seq_preds = np.array(model.crf.decode(seq_logits))
                else:
                    seq_preds = seq_logits.detach().cpu().numpy()
                all_seq_label_mask = batch[3].detach().cpu().numpy()
            else:
                if args.use_crf:
                    seq_preds = np.append(seq_preds, np.array(model.crf.decode(seq_logits)), axis=0)
                else:
                    seq_preds = np.append(seq_preds, seq_logits.detach().cpu().numpy(), axis=0)
                all_seq_label_mask = np.append(all_seq_label_mask, batch[3].detach().cpu().numpy(), axis=0)
    if not args.use_crf:
        seq_preds = np.argmax(seq_preds, axis=2)

    seq_label_map = {i: label for i, label in enumerate(seq_label_lst)}
    # 修改部分label结果
    seq_label_map[0] = 'O'
    seq_label_map[1] = 'O'
    seq_preds_list = [[] for _ in range(seq_preds.shape[0])]

    for i in range(seq_preds.shape[0]):
        for j in range(seq_preds.shape[1]):
            if all_seq_label_mask[i, j] != pad_token_label_id:
                seq_preds_list[i].append(seq_label_map[seq_preds[i][j]])

    assert len(seq_preds_list) == len(eids) == len(sids)
    # 保存文件
    outputs = {}
    for i in range(len(seq_preds_list)):
        pred_seq = seq_preds_list[i]
        eid = eids[i]
        sid = sids[i]
        if eid not in outputs:
            outputs[eid] = {}
            outputs[eid][sid] = ' '.join(pred_seq[3:])  # 只保留句子的BIO标签，删去了speaker的BIO标签
        else:
            outputs[eid][sid] = ' '.join(pred_seq[3:])

    pred_path = os.path.join(pred_config.test_output_file)
    with open(pred_path, 'w', encoding='utf-8') as json_file:
        json.dump(outputs, json_file, ensure_ascii=False, indent=4)

    logger.info("Prediction Done!")


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_input_file", default="../../dataset/test.json", type=str, help="Input file for prediction")
    parser.add_argument("--test_output_file", default="submission_track1_task1.json", type=str, help="Output file for prediction")
    parser.add_argument("--model_dir", default="./save_model", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    pred_config = parser.parse_args()
    predict(pred_config)
