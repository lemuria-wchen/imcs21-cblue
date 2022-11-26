## IMCS-NER Task

本文档包含 **医疗命名实体识别（IMCS-NER）** 任务的：

- 任务定义
- 提交格式
- 评价方法
- 基线模型

### 任务定义

**IMCS-NER** 任务的目标是从医患对话文本中 *识别出医疗相关的命名实体*，该任务的输入输出示例如下。

| Input（对话）                  | Output （BIO标签）                                                                             |
|----------------------------|:-------------------------------------------------------------------------------------------|
| ...                        | ...                                                                                        |
| 医生：有没有**发热**               | O O O O O O O B-Symptom I-Symptom                                                          |
| 患者：没有                      | O O O O O                                                                                  |
| ...                        | ...                                                                                        |
| 医生：用过什么药物                  | O O O O O O O O O                                                                          |
| 患者：给喝过**小儿咳喘灵**，**阿莫西林颗粒** | O O O O O O B-Drug I-Drug I-Drug I-Drug I-Drug O B-Drug I-Drug I-Drug I-Drug I-Drug I-Drug |
| ...                        | ...                                                                                        |

### 提交格式

**IMCS-NER** 任务要求提交文件为 **json** 格式，具体格式如下。

```
{
    "example_id1": {    # 测试集样本id
        "sentence_id1": "O O O O O O O O O O O", 
        "sentence_id2": "O O O O O B-Symptom I-Symptom O O O O O O", 
        ...
    }, 
    "example_id2":{
   	...
    }
...
}
```

### 评价方法

**IMCS-NER** 任务采用实体级的 **F1 score** 作为评价指标，详细见文件 `eval_task1.py`, 运行方式如下：

```shell
python3 eval_task1.py --gold_path {gold_file_path} --pred_path {pred_file_path}
```

### 基线模型

我们为 **IMCS-NER** 任务创建了 2 个基线模型，详情如下。

| Model      | F1-score (%) | Link                                                                     |
|------------|--------------|--------------------------------------------------------------------------|
| LSTM-NER   | 86.02        | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task1/LSTM-NER   |
| BERT-NER   | 87.30        | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task1/BERT-NER   |

