import os
import pandas as pd 
import numpy as np
from pandas.core.frame import DataFrame
from sumeval.metrics.rouge import RougeCalculator
def evaluate_from_decoded(ref_dir,dec_dir,word2id):
    scores = []
    lens = []
    f_names = os.listdir(ref_dir)
    nums = len(os.listdir(ref_dir))

    for i in range(nums):
        # id_ =  str(i).zfill(3)
        try:
            with  open(os.path.join(dec_dir,f_names[i]),encoding='utf-8') as fd: 
                sh = fd.readlines()[-1]
                lens.append(len(sh))
        

            srs = []
            #2个参考诊疗报告
            with open(os.path.join(ref_dir,f_names[i]),encoding='utf-8') as fr: 
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
                    else: # OOV
    #                     print(w)
                        str_h += '-1'
                    str_h += ' '

            score0 = compute_scores(str_h,str_srs)
            scores.append(score0)
            print(score0)
        except Exception as e:
            print(e)
            scores.append([0.0,0.0,0.0])

    df_scores = DataFrame(scores)
    mean_scores = [round(df_scores[i].mean(),4) for i in range(3)]
    print('ROUGE-1:',mean_scores[0],'\nROUGE-2:',mean_scores[1],'\nROUGE-L:',mean_scores[2])
    return mean_scores

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


if __name__ == '__main__':   
    with open('./vocab',encoding='utf-8') as f:
        word2id = {}
        while True:
            l = f.readline()
            if not l:
                break
            a = l.split(' ')[0]
            i = l.split(' ')[1]
            word2id[a] = str(i) 
    mean_scores = evaluate_from_decoded('参考摘要_test','生成摘要_test',word2id)
