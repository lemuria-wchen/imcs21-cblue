import os
import pandas as pd 
import numpy as np
from pandas.core.frame import DataFrame
import logging
import tensorflow as tf
from sumeval.metrics.rouge import RougeCalculator

def compute_scores(sh,srs):
    rouge = RougeCalculator(stopwords=False, lang="en")
    rouge_1 = rouge.rouge_n(
            summary=sh,
            references=srs,
            n=1)
    rouge_2 = rouge.rouge_n(
            summary=sh,
            references=srs,
            n=2)
    rouge_l = rouge.rouge_l(
            summary=sh,
            references=srs)
    return [rouge_1, rouge_2, rouge_l]


def evaluate_from_decoded(ref_dir,dec_dir,word2id):
    '''通过自动化的方式计算rouge值
    # 对于测试集测评时也是使用该方法'''
    scores = []
    lens = []
    f_names = os.listdir(dec_dir)
    nums = len(os.listdir(dec_dir))

    for i in range(nums):
        id_ =  str(i).zfill(3)
        with  open(os.path.join(dec_dir,f_names[i])) as fd: 
            sh = fd.readlines()[-1]
            lens.append(len(sh))

        srs = []
        #2个参考诊疗报告
        with open(os.path.join(ref_dir,f_names[i])) as fr: 
            sr = fr.readlines()
            sr_0 = sr[:6]
            sr_1 = sr[6:]
            
            tmp_data = [x[:-1] for x in sr_0]
            sr_0 = ''.join(tmp_data)
            sr_0 = sr_0+'。'
            srs.append(sr_0)
            
            tmp_data = [x[:-1] for x in sr_1]
            sr_1 = ''.join(tmp_data)
            sr_1 = sr_1+'。'
            srs.append(sr_1)

            # word2id
            str_srs = []
            for sr in srs:
                str_r = ''
                sr_split = sr.split(' ')
                for w in sr_split:
                    if w in word2id:
                        str_r += str(word2id[w])
                    else:
                        str_r += '-1'
                    str_r += ' '

                str_srs.append(str_r)

            #对生成文本进行word2id的转换
            str_h = ''
            sh_split = sh.split(' ')
            for w in sh_split:
                if w in word2id:
                    str_h += str(word2id[w])
                else:
                    print(w)
                    str_h += '-1'
                str_h += ' '

        score0 = compute_scores(str_h,str_srs)
        scores.append(score0)

    df_scores = DataFrame(scores)
    mean_scores = [round(df_scores[i].mean(),4) for i in range(3)]
    return mean_scores


def rouge_log(mean_scores, dir_to_write):
    '''
    将ROUGE值记录在'ROUGE_results.txt'中
    '''
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    print("ROUGE值记录在 %s 中"%(results_file))
    with open(results_file, "w") as f:
        f.write('计算出验证集的R-1，R-2，R-l结果为：\n')
        f.write(str(mean_scores[0])+'\t'+str(mean_scores[1])+'\t'+str(mean_scores[2]))


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
    '''计算误差'''
    if running_avg_loss == 0:  
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    return running_avg_loss


def write_for_rouge(reference_sents, decoded_words, f_name,
                        _rouge_ref_dir, _rouge_dec_dir,compute_rouge):
    '''将生产的诊疗报告f_name文件中，为后续计算ROUGE做准备
    # 如果是对验证集进行操作，则可以设置compute_rouge为True，将参考摘要写入_rouge_ref_dir文件夹中。
    '''
    decoded_sents = []
    while len(decoded_words) > 0:
        try:
            fst_period_idx = decoded_words.index(".")
        except ValueError:
            fst_period_idx = len(decoded_words)
        sent = decoded_words[:fst_period_idx + 1]
        decoded_words = decoded_words[fst_period_idx + 1:]
        decoded_sents.append(' '.join(sent))

    if compute_rouge==True:
        ref_file = os.path.join(_rouge_ref_dir, f_name)
        with open(ref_file, "w") as f:
            for idx, sent in enumerate(reference_sents):
                f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")

    decoded_file = os.path.join(_rouge_dec_dir, f_name)
    with open(decoded_file, "w") as f:
        for idx, sent in enumerate(decoded_sents):
            f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")

    print("Wrote example {} to file".format(f_name))
