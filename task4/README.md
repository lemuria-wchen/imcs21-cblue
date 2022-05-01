## IMCS-MRG Task

本文档包含 **对话意图识别（IMCS-DAC）** 任务的：

- 任务定义
- 提交格式
- 评价方法
- 基线模型

### 任务定义

**IMCS-MGR** 任务的目标是自动从话语中 **识别医生或者患者的意图**，该任务的输入输出示例如下。

| Input（对话中的某个话语）             | Output（意图）      |
|-----------------------------|-----------------|
| 医生：你好，咳嗽是连声咳吗？有痰吗？有没流鼻涕，鼻塞？ | Request-Symptom |


### 提交格式

**IMCS-MGR** 任务要求提交文件为 **json** 格式，具体格式如下。

```
{
    "example_id1": {    # 测试集样本id
        "sentence_id1": "Request-Symptom", 
        "sentence_id2": "Inform-Symptom", 
        ...
    }, 
    "example_id2":{
   	...
    }
    ...
}
```

### 评价方法

**IMCS-DAC** 任务采用准确率 **Acc** 作为评价指标，详细见文件 `eval_task4.py`, 运行方式如下：

```shell
python3 eval_task4.py --gold_path {gold_file_path} --pred_path {pred_file_path}
```

### 基线模型

我们为 **IMCS-DAC** 任务创建了 5 个基线模型，模型详情如下。

| Model    | Acc    | Link                                                                   |
|----------|--------|------------------------------------------------------------------------|
| TextCNN  | 0.7893 | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task4/DNN-DAC  |
| TextRNN	 | 0.7846 | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task4/DNN-DAC  |
| TextRCNN | 0.7949 | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task3/DNN-DAC  |
| DPCNN	   | 0.7791 | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task3/DNN-DAC  |
| BERT	    | 0.8165 | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task3/BERT-DAC |
| ERNIE	   | 0.8191 | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task3/BERT-DAC |

