import sys
import os
import struct
import subprocess
import collections
import json
import pandas as pd
import tensorflow as tf
from tensorflow.core.example import example_pb2

import argparse


# 使用SENTENCE_START,SENTENCE_END划分句子
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

# END_TOKENS中的字符，是一个句子结束时可能用到的字符。如果句子不以此结尾，则添加“。”作为句子结尾。
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', ")",'。','！','？','）','>','，']


def get_tokenized_data(medi_tokenized_dir,df_id_report):
    '''将来自df_id_report中的数据进行处理，将字符串切分，存在文件夹 medi_tokenized_dir 中。
    # 因为每一个example_id有两个不同的参考诊疗报告。所以训练集、验证集和测试集的处理方式不同。
    # 训练集中，每组对话对应1份参考诊疗报告，可能存在两个相同的输入对应两个不同的输出。
    # 验证集中，每组对话对应2份参考诊疗报告。自动化评价中，需要同时使用2份参考诊疗报告，对1组对话的生成诊疗报告评分。
    # 测试集中，不存在参考诊疗报告。
    '''
    for i in range(len(df_id_report)):
        id_ = df_id_report['example_id'][i]
        self_report = df_id_report['self_report'][i]
        content = df_id_report['dialogue'][i]
        title = [df_id_report['report1'][i], df_id_report['report2'][i]]
        split = df_id_report['split'][i]
        
        if split=='train':
            for j in range(2):
                with open(os.path.join(medi_tokenized_dir,split+'_'+str(id_)+'_'+str(j)), 'w',encoding='utf-8') as f:
                    f.write('患 者 ： '+' '.join([s for s in self_report]))
                    tmp_content = content[:-1].split('||')
                    for i,sentence in enumerate(tmp_content):
                        if len(sentence)==0:
                            tmp_content[i] = '。'
                        elif sentence[-1] not in END_TOKENS:
                            tmp_content[i] = sentence+'。'
                        f.write('\n\n'+' '.join([s for s in tmp_content[i]]))
                    tmp_title = title[j].strip().split('\n')
                    for sentence in tmp_title:
                        f.write('\n\n@highlight\n\n')
                        if sentence[-1] not in END_TOKENS:
                            f.write(' '.join([s for s in sentence])+' 。')
                        else:
                            f.write(' '.join([s for s in sentence]))
        elif split == 'dev':
            with open(os.path.join(medi_tokenized_dir,split+'_'+str(id_)), 'w',encoding='utf-8') as f:
                    f.write('患 者 ： '+' '.join([s for s in self_report]))
                    tmp_content = content[:-1].split('||')
                    for i,sentence in enumerate(tmp_content):
                        if len(sentence)==0:
                            tmp_content[i] = '。'
                        elif sentence[-1] not in END_TOKENS:
                            tmp_content[i] = sentence+'。'
                        f.write('\n\n'+' '.join([s for s in tmp_content[i]]))
                    tmp_title = title[0].strip().split('\n')
                    for sentence in tmp_title:
                        f.write('\n\n@highlight\n\n')
                        if sentence[-1] not in END_TOKENS:
                            f.write(' '.join([s for s in sentence])+' 。')
                        else:
                            f.write(' '.join([s for s in sentence]))
                    tmp_title = title[1].strip().split('\n')
                    for sentence in tmp_title:
                        f.write('\n\n@highlight\n\n')
                        if sentence[-1] not in END_TOKENS:
                            f.write(' '.join([s for s in sentence])+' 。')
                        else:
                            f.write(' '.join([s for s in sentence]))
        else:
            with open(os.path.join(medi_tokenized_dir,split+'_'+str(id_)), 'w',encoding='utf-8') as f:
                f.write('患 者 ： '+' '.join([s for s in self_report]))
                tmp_content = content[:-1].split('||')
                for i,sentence in enumerate(tmp_content):
                    if len(sentence)==0:
                            tmp_content[i] = '。'
                    elif sentence[-1] not in END_TOKENS:
                        tmp_content[i] = sentence+'。'
                    f.write('\n\n'+' '.join([s for s in tmp_content[i]]))
                f.write('\n\n@highlight\n\n')


def chunk_file(set_name):
    '''数据量较大时，可以对数据进行切分'''
    in_file = os.path.join(medi_finished_dir, "%s.bin" % set_name)  
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) 
        with open(chunk_fname, 'wb') as writer:
        # with open(chunk_fname, 'w') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def read_text_file(text_file):
    lines = []
    with open(text_file, "r",encoding='UTF-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def fix_missing_period(line):
    """部分句子没有END_TOKEN，如果没有，就进行添加。"""
    if "@highlight" in line: return line
    if line=="": return line
    if line[-1] in END_TOKENS: return line
    return line + " 。"


def get_dia_abs(dialogue_file):
    '''从文件中获取对话和诊疗报告'''
    lines = read_text_file(dialogue_file)
    lines = [fix_missing_period(line) for line in lines]

    dialogue_lines = []
    highlights = []
    next_is_highlight = False
    for idx,line in enumerate(lines):
        if line == "":
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            dialogue_lines.append(line)

    dialogue = ' '.join(dialogue_lines)
    # 用<s> 和 </s> 区分诊疗报告中的6个部分。
    abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])
    return dialogue, abstract


def write_to_bin(mode, in_file, out_file, makevocab=False):
    '''二进制处理'''
    dias_fnames = os.listdir(in_file)
    dialogue_fnames = []
    for i in dias_fnames:
        if i.startswith(mode):
            dialogue_fnames.append(i)
    print("开始 %s 数据的二进制转换" % mode)
    num_dias = len(dialogue_fnames)
    print(mode,'对话数量:',num_dias)

    if makevocab:
        vocab_counter = collections.Counter()
    with open(os.path.join(medi_finished_dir,'file_names_'+mode),'w') as ff:
        with open(out_file, 'wb') as writer:
            for idx,s in enumerate(dialogue_fnames):
                if idx % 500 == 0:
                    print("写入第 %i 个对话; 已完成 %.2f %% " % (idx, float(idx)*100.0/float(num_dias)))

                dialogue_file = os.path.join(in_file, s)
                dialogue, abstract = get_dia_abs(dialogue_file)

                tf_example = example_pb2.Example()
                tf_example.features.feature['dialogue'].bytes_list.value.extend([dialogue.encode()])
                tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
                tf_example_str = tf_example.SerializeToString()
                str_len = len(tf_example_str)
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, tf_example_str))
                ff.write(s+'\n')

                if makevocab:
                    dia_tokens = dialogue.split(' ')
                    abs_tokens = abstract.split(' ')
                    abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] 
                    tokens = dia_tokens + abs_tokens
                    tokens = [t.strip() for t in tokens] 
                    tokens = [t for t in tokens if t!=""] 
                    vocab_counter.update(tokens)

    print("文件写入 %s\n" % out_file)

    # 创建字典
    if makevocab:
        with open(os.path.join(medi_finished_dir, "vocab"), 'w', encoding='utf-8') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("完成字典 vocab 的创建和保存\n")

# 对json文件进行一次处理
def deal_with_json():
    f_train = './dataset/train.json'
    f_test = './dataset/test.json'
    with open(f_train,'r',encoding='utf8') as fr:
        data_train = json.load(fr)
    with open(f_test,'r',encoding='utf8') as fr:
        data_test = json.load(fr)
    eids_train = data_train.keys()
    eids_test = data_test.keys()
    df_split = pd.read_csv('./dataset/split.csv')
    id_report = []
    for eid in eids_train:
        self_report = data_train[eid]['self_report']
        dialogue_list = [x['sentence'] for x in data_train[eid]['dialogue']]
        dialogue = '||'.join(dialogue_list)
        report1 = data_train[eid]['report'][0]
        report2 = data_train[eid]['report'][1]
        split = df_split[df_split['example_id']==int(eid)]['split'].values[0]
        id_report.append([eid,self_report,dialogue,report1,report2,split])
    for eid in eids_test:
        self_report = data_test[eid]['self_report']
        dialogue_list = [x['sentence'] for x in data_test[eid]['dialogue']]
        dialogue = '||'.join(dialogue_list)
        report1 = 'None'
        report2 = 'None'
        split = 'test'
        id_report.append([eid,self_report,dialogue,report1,report2,split])
    df_id_report = pd.DataFrame(id_report,columns=['example_id','self_report','dialogue','report1','report2','split'])
    df_id_report.to_csv('./example_report.csv',index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument("--vocab_size", default=3000, type=int ,
                        help="词典大小")
    parser.add_argument("--chunk_size", default=500, type=int , 
                        help="chunk数据时，切分的大小")
    parser.add_argument("--example_report_file", default='example_report.csv', type=str, 
                        help="原始数据")
    parser.add_argument("--medi_tokenized_dir", default='medi_tokenized_dir', type=str, 
                        help="toknized之后的文件夹路径")
    parser.add_argument("--medi_finished_dir", default='medi_finished_dir', type=str, 
                        help="数据预处理完成后的文件夹路径")
    args = parser.parse_args()

    deal_with_json()
    VOCAB_SIZE = args.vocab_size
    CHUNK_SIZE = args.chunk_size
    example_report_file = args.example_report_file
    medi_tokenized_dir = args.medi_tokenized_dir
    medi_finished_dir = args.medi_finished_dir
    if not os.path.isdir(medi_tokenized_dir):
        os.mkdir(medi_tokenized_dir)
    if not os.path.isdir(medi_finished_dir):
        os.mkdir(medi_finished_dir)
    
    df_id_report = pd.read_csv(example_report_file)
    get_tokenized_data(medi_tokenized_dir,df_id_report)

    write_to_bin("test", medi_tokenized_dir,os.path.join(medi_finished_dir, "test.bin"))
    write_to_bin("dev", medi_tokenized_dir,os.path.join(medi_finished_dir, "dev.bin"))
    write_to_bin("train", medi_tokenized_dir,os.path.join(medi_finished_dir, "train.bin"), makevocab=True)

    chunks_dir = os.path.join(medi_finished_dir, "chunked")
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    chunk_file('train')
    # print("Saved chunked data in %s" % chunks_dir)

