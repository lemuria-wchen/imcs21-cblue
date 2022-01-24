import random
import numpy as np
import json

            
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


class DataProcessor_MTL_LSTM(object):
    """训练时数据处理"""
    def __init__(self, 
                 input_seq_path, 
                 output_seq_bio_path,
                 output_seq_attr_path,
                 output_seq_type_path,
                 w2i_char,
                 w2i_bio,
                 w2i_attr,
                 w2i_type,
                 shuffling=False):
        
        inputs_seq = []
        with open(input_seq_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                seq = [w2i_char[word] if word in w2i_char else w2i_char["[UNK]"] for word in line.split(" ")]
                inputs_seq.append(seq)
                
        outputs_seq_bio = []
        with open(output_seq_bio_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                seq = [w2i_bio[word] for word in line.split(" ")]
                outputs_seq_bio.append(seq)
        
        outputs_seq_attr = []
        with open(output_seq_attr_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                seq = [w2i_attr[word] for word in line.split(" ")]
                outputs_seq_attr.append(seq)

        outputs_seq_type = []
        with open(output_seq_type_path, "r", encoding="utf-8") as f:
            for line in f.read().strip().split("\n"):
                seq = [w2i_type[word] for word in line.split(" ")]
                outputs_seq_type.append(seq)

        assert len(inputs_seq) == len(outputs_seq_bio)
        assert all(len(input_seq) == len(output_seq_bio) for input_seq, output_seq_bio in zip(inputs_seq, outputs_seq_bio))
        assert len(inputs_seq) == len(outputs_seq_attr)
        assert all(len(input_seq) == len(output_seq_attr) for input_seq, output_seq_attr in zip(inputs_seq, outputs_seq_attr))
        assert len(inputs_seq) == len(outputs_seq_type)
        assert all(len(input_seq) == len(output_seq_type) for input_seq, output_seq_type in zip(inputs_seq, outputs_seq_type))

        self.w2i_char = w2i_char
        self.w2i_bio = w2i_bio
        self.w2i_attr = w2i_attr
        self.w2i_type = w2i_type
        self.inputs_seq = inputs_seq
        self.outputs_seq_bio = outputs_seq_bio
        self.outputs_seq_attr = outputs_seq_attr
        self.outputs_seq_type = outputs_seq_type
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
        outputs_seq_bio_batch = []
        outputs_seq_attr_batch = []
        outputs_seq_type_batch = []
        
        while (len(inputs_seq_batch) < batch_size) and (not self.end_flag):
            p = self.ps[self.pointer]
            inputs_seq_batch.append(self.inputs_seq[p].copy())
            inputs_seq_len_batch.append(len(self.inputs_seq[p]))
            outputs_seq_bio_batch.append(self.outputs_seq_bio[p].copy())
            outputs_seq_attr_batch.append(self.outputs_seq_attr[p].copy())
            outputs_seq_type_batch.append(self.outputs_seq_type[p].copy())
            self.pointer += 1
            if self.pointer >= len(self.ps): self.end_flag = True
        
        max_seq_len = max(inputs_seq_len_batch)
        for seq in inputs_seq_batch:
            seq.extend([self.w2i_char["[PAD]"]] * (max_seq_len - len(seq)))
        for seq in outputs_seq_bio_batch:
            seq.extend([self.w2i_bio["O"]] * (max_seq_len - len(seq)))
        for seq in outputs_seq_attr_batch:
            seq.extend([self.w2i_attr["null"]] * (max_seq_len - len(seq)))
        for seq in outputs_seq_type_batch:
            seq.extend([self.w2i_type["null"]] * (max_seq_len - len(seq)))

        return (np.array(inputs_seq_batch, dtype="int32"),
                np.array(inputs_seq_len_batch, dtype="int32"),
                np.array(outputs_seq_bio_batch, dtype="int32"),
                np.array(outputs_seq_attr_batch, dtype="int32"),
                np.array(outputs_seq_type_batch, dtype="int32"))


class DataProcessor_MTL_LSTM_Test(object):
    """测试时数据处理"""
    def __init__(self,
                 input_path,
                 w2i_char,
                 w2i_bio,
                 w2i_attr,
                 w2i_type,
                 shuffling=False):

        inputs_seq = []
        eids = []
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for k, v in data.items():
            for sent in data[k]['dialogue']:
                words = list(sent['speaker'] + '：' + sent['sentence'])
                seq = [w2i_char[word] if word in w2i_char else w2i_char["[UNK]"] for word in words]
                inputs_seq.append(seq)
                eids.append(k)

        self.w2i_char = w2i_char
        self.w2i_bio = w2i_bio
        self.w2i_attr = w2i_attr
        self.w2i_type = w2i_type
        self.inputs_seq = inputs_seq
        self.eids = eids
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
        eids_batch = []

        while (len(inputs_seq_batch) < batch_size) and (not self.end_flag):
            p = self.ps[self.pointer]
            inputs_seq_batch.append(self.inputs_seq[p].copy())
            inputs_seq_len_batch.append(len(self.inputs_seq[p]))
            eids_batch.append(self.eids[p])
            self.pointer += 1
            if self.pointer >= len(self.ps): self.end_flag = True

        max_seq_len = max(inputs_seq_len_batch)
        for seq in inputs_seq_batch:
            seq.extend([self.w2i_char["[PAD]"]] * (max_seq_len - len(seq)))

        return (np.array(inputs_seq_batch, dtype="int32"),
                np.array(inputs_seq_len_batch, dtype="int32"),
                np.array(eids_batch, dtype=str))


def extract_kvpairs_in_bioes_type(bio_seq, word_seq, attr_seq, type_seq):
    assert len(bio_seq) == len(word_seq) == len(attr_seq) == len(type_seq)
    pairs = set()
    v = ""
    for i in range(len(bio_seq)):
        word = word_seq[i]
        bio = bio_seq[i]
        attr = attr_seq[i]
        type = type_seq[i]

        if bio == "O":
            v = ""
        elif bio == "S":
            v = word
            pairs.add((attr, type, v))
            v = ""
        elif bio == "B":
            v = word
        elif bio == "I":
            if v != "":
                v += word
        elif bio == "E":
            if v != "":
                v += word
                pairs.add((attr, type, v))
            v = ""
    return pairs


def extract_kvpairs_in_bio_type(bio_seq, word_seq, attr_seq, type_seq):
    assert len(bio_seq) == len(word_seq) == len(attr_seq) == len(type_seq)
    pairs = set()
    v = ""
    for i in range(len(bio_seq)):
        word = word_seq[i]
        bio = bio_seq[i]

        if bio == "O":
            if v != "":
                pairs.add((attr_seq[i - 1], type_seq[i - 1], v))
            v = ""
        elif bio == "B":
            if v != "":
                pairs.add((attr_seq[i - 1], type_seq[i - 1], v))
            v = word
        elif bio == "I":
            if v != "":
                v += word
    if v != "":
        pairs.add((attr_seq[-1], type_seq[-1], v))

    return pairs


def cal_f1_score(preds, golds):
    """评价指标，注意其与eval_track1_task2.py中的区别"""
    assert len(preds) == len(golds)
    p_sum = 0
    r_sum = 0
    hits = 0
    for pred, gold in zip(preds, golds):
        p_sum += len(pred)
        r_sum += len(gold)
        for label in pred:
            if label in gold:
                hits += 1
    p = hits / p_sum if p_sum > 0 else 0
    r = hits / r_sum if r_sum > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1
