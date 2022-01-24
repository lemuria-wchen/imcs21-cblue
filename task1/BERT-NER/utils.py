import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

from transformers import BertConfig
from transformers import BertTokenizer

from modeling_nerbert import NERBERT

# 模型类别
MODEL_CLASSES = {
    'bert': (BertConfig, NERBERT, BertTokenizer),
}

# 模型路径
MODEL_PATH_MAP = {
    'bert': 'bert-base-chinese',
}

def get_seq_labels(args):
    """获取序列标签"""
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.seq_label_file), 'r', encoding='utf-8')]

def load_tokenizer(args):
    """加载rokenizer"""
    # 在线下载
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)
    # # 从已下载文件中读取
    # return MODEL_CLASSES[args.model_type][2].from_pretrained('./bert-base-chinese/bert-base-chinese-vocab.txt')

def init_logger():
    """logger初始化"""
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    """随机种子设置"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(seq_preds, seq_labels):
    """获得评价结果"""
    assert len(seq_preds) == len(seq_labels)
    results = {}
    seq_result = get_seq_metrics(seq_preds, seq_labels)

    results.update(seq_result)

    return results


def get_seq_metrics(preds, labels):
    """计算评价结果"""
    assert len(preds) == len(labels)
    return {
        "seq_precision": precision_score(labels, preds),
        "seq_recall": recall_score(labels, preds),
        "seq_f1": f1_score(labels, preds)
    }



def read_prediction_text(args):
    """读取预测文本"""
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]
