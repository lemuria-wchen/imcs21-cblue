import random
import numpy as np
import json
from seqeval.metrics import precision_score, recall_score, f1_score

            
def load_vocabulary(path):
    """生成辅助字典"""
    vocab = open(path, "r", encoding="utf-8").read().strip().split("\n")
    print("load vocab from: {}, containing words: {}".format(path, len(vocab)))
    w2i = {}
    i2w = {}
    for i, w in enumerate(vocab):
        w2i[w] = i
        i2w[i] = w
    return w2i, i2w


class DataProcessor_LSTM(object):
    """训练时数据处理"""
    def __init__(self, 
                 input_seq_path, 
                 output_seq_path, 
                 w2i_char,
                 w2i_bio,
                 shuffling=False):
        
        inputs_seq = []
        with open(input_seq_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                seq = [w2i_char[word] if word in w2i_char else w2i_char["[UNK]"] for word in line.split(" ")]
                inputs_seq.append(seq)
                
        outputs_seq = []
        with open(output_seq_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                seq = [w2i_bio[word] for word in line.split(" ")]
                outputs_seq.append(seq)
                    
        assert len(inputs_seq) == len(outputs_seq)
        assert all(len(input_seq) == len(output_seq) for input_seq, output_seq in zip(inputs_seq, outputs_seq))
        
        self.w2i_char = w2i_char
        self.w2i_bio = w2i_bio
        self.inputs_seq = inputs_seq
        self.outputs_seq = outputs_seq
        self.ps = list(range(len(inputs_seq)))
        self.shuffling = shuffling
        if shuffling: random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False
        print("DataProcessor load data num: " + str(len(inputs_seq)), "shuffling:", shuffling)
        
    def refresh(self):
        if self.shuffling: random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False
    
    def get_batch(self, batch_size):
        inputs_seq_batch = []
        inputs_seq_len_batch = []
        outputs_seq_batch = []
        
        while (len(inputs_seq_batch) < batch_size) and (not self.end_flag):
            p = self.ps[self.pointer]
            inputs_seq_batch.append(self.inputs_seq[p].copy())
            inputs_seq_len_batch.append(len(self.inputs_seq[p]))
            outputs_seq_batch.append(self.outputs_seq[p].copy())
            self.pointer += 1
            if self.pointer >= len(self.ps): self.end_flag = True
        
        max_seq_len = max(inputs_seq_len_batch)
        for seq in inputs_seq_batch:
            seq.extend([self.w2i_char["[PAD]"]] * (max_seq_len - len(seq)))
        for seq in outputs_seq_batch:
            seq.extend([self.w2i_bio["O"]] * (max_seq_len - len(seq)))
        
        return (np.array(inputs_seq_batch, dtype="int32"),
                np.array(inputs_seq_len_batch, dtype="int32"),
                np.array(outputs_seq_batch, dtype="int32"))



class DataProcessor_LSTM_Test(object):
    """测试时数据处理"""
    def __init__(self,
                 input_path,
                 w2i_char,
                 w2i_bio,
                 shuffling=False):

        inputs_seq = []
        eids = []
        sids = []
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for k, v in data.items():
            for sent in data[k]['dialogue']:
                words = list(sent['speaker'] + '：' + sent['sentence'])
                seq = [w2i_char[word] if word in w2i_char else w2i_char["[UNK]"] for word in words]
                inputs_seq.append(seq)
                eids.append(k)
                sids.append(sent['sentence_id'])

        self.w2i_char = w2i_char
        self.w2i_bio = w2i_bio
        self.inputs_seq = inputs_seq
        self.eids = eids
        self.sids = sids
        self.ps = list(range(len(inputs_seq)))
        self.shuffling = shuffling
        if shuffling: random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False
        print("DataProcessor load data num: " + str(len(inputs_seq)))

    def refresh(self):
        if self.shuffling: random.shuffle(self.ps)
        self.pointer = 0
        self.end_flag = False

    def get_batch(self, batch_size):
        inputs_seq_batch = []
        inputs_seq_len_batch = []
        eids_batch = []
        sids_batch = []

        while (len(inputs_seq_batch) < batch_size) and (not self.end_flag):
            p = self.ps[self.pointer]
            inputs_seq_batch.append(self.inputs_seq[p].copy())
            inputs_seq_len_batch.append(len(self.inputs_seq[p]))
            eids_batch.append(self.eids[p])
            sids_batch.append(self.sids[p])
            self.pointer += 1
            if self.pointer >= len(self.ps): self.end_flag = True

        max_seq_len = max(inputs_seq_len_batch)
        for seq in inputs_seq_batch:
            seq.extend([self.w2i_char["[PAD]"]] * (max_seq_len - len(seq)))

        return (np.array(inputs_seq_batch, dtype="int32"),
                np.array(inputs_seq_len_batch, dtype="int32"),
                np.array(eids_batch, dtype=str),
                np.array(sids_batch, dtype=str))


def compute_metrics(seq_preds, seq_golds):
    """获得评价结果"""
    assert len(seq_preds) == len(seq_golds)
    results = {}
    seq_result = get_metrics(seq_preds, seq_golds)

    results.update(seq_result)

    return results


def get_metrics(preds, golds):
    """计算评价结果"""
    assert len(preds) == len(golds)
    return {
        "precision": precision_score(golds, preds),
        "recall": recall_score(golds, preds),
        "f1": f1_score(golds, preds)
    }
