import json
import sys
from pandas.core.frame import DataFrame
from sumeval.metrics.rouge import RougeCalculator


def make_vocab(path):
    vocab = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word, word_id = line.split()
            vocab[word] = str(word_id)
    return vocab


def tokenize(report: str, vocab: dict):
    ids = []
    for word in report.replace(' ', '').replace('\n', ''):
        if word in vocab:
            ids.append(str(vocab.get(word)))
        else:
            ids.append(str(-1))
    return ' '.join(ids)


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def compute_scores(summary, references):
    rouge = RougeCalculator(stopwords=False, lang="en")
    rouge_1 = rouge.rouge_n(summary=summary, references=references, n=1)
    rouge_2 = rouge.rouge_n(summary=summary, references=references, n=2)
    rouge_l = rouge.rouge_l(summary=summary, references=references)
    return rouge_1, rouge_2, rouge_l


def evaluate(golds, preds, vocab):
    assert len(golds) == len(preds)
    scores = []
    for key, val in preds.items():
        assert key in golds
        gold_reports = [tokenize(report, vocab) for report in golds[key]['report']]
        pred_report = tokenize(val, vocab)
        try:
            score = compute_scores(pred_report, gold_reports)
            scores.append(score)
        except Exception as e:
            print(e)
            scores.append([0.0, 0.0, 0.0])
    df_scores = DataFrame(scores)
    mean_scores = [round(df_scores[i].mean(), 4) for i in range(3)]
    return {'ROUGE-1': mean_scores[0], 'ROUGE-2': mean_scores[1], 'ROUGE-L': mean_scores[2]}


if __name__ == "__main__":
    gold_data = load_json(sys.argv[1])  # 读入test的真实数据
    pred_data = load_json(sys.argv[2])  # 读入test的预测数据
    word2id = make_vocab('task3/vocab')
    print(evaluate(gold_data, pred_data, word2id))
