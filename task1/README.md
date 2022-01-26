## IMCS-NER Task

### 任务定义

本任务为医疗命名实体识别（**IMCS-NER**）任务，即自动从医患对话文本中识别出重要的实体，共有五类医疗相关实体，采用字符级别的 *BIO* 标注体系进行标注。**IMCS-NER** 任务的输入输出实例如下。

| Input（对话）                  | Output （BIO标签）                                                                             |
|----------------------------|:-------------------------------------------------------------------------------------------|
| ...                        | ...                                                                                        |
| 医生：有没有**发热**               | O O O O O O O B-Symptom I-Symptom                                                          |
| 患者：没有                      | O O O O O                                                                                  |
| ...                        | ...                                                                                        |
| 医生：用过什么药物                  | O O O O O O O O O                                                                          |
| 患者：给喝过**小儿咳喘灵**，**阿莫西林颗粒** | O O O O O O B-Drug I-Drug I-Drug I-Drug I-Drug O B-Drug I-Drug I-Drug I-Drug I-Drug I-Drug |
| ...                        | ...                                                                                        |


### 提交文件格式

**IMCS-NER** 任务要求提交文件为 **json** 格式，一个样例提交文件可以在 [这里]() 找到，具体格式如下。

```
{
    "example_id1": {    # 测试集样本id
        "sentence_id1": "O O O O O O O O O O O",  
        "sentence_id2": "O O O O O B-Symptom I-Symptom O O O O O O",
        ...
    }
   	"example_id2":{
   	...
   	}
...
}
```

### 评价方式

**IMCS-NER** 任务采用 **F1 score** 作为评价指标，详细见文件 `eval_task1.py`, 运行方式如下：

```
python eval_task1.py {gold_data_path} {pred_data_path}
```

### 基线模型

我们为 **IMCS-NER** 任务创建了 2 个基线模型，模型详情如下。

| Model    | F1-score (%) | Link                                                                   |
|----------|--------------|------------------------------------------------------------------------|
| LSTM-NER | 92.21        | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task1/LSTM-NER |
| BERT-NER | 91.66        | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task1/BERT-NER |

