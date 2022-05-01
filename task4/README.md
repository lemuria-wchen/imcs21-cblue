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

**IMCS-MGR** 任务采用 **Rouge** 作为评价指标，详细见文件 `eval_task3.py`, 运行方式如下：

```shell
python3 eval_task3.py --gold_path {gold_file_path} --pred_path {pred_file_path}
```

### 基线模型

我们为 **IMCS-MGR** 任务创建了 5 个基线模型，模型详情如下。

| Model              | Avg. Rouge-score | Link                                                                     |
|--------------------|------------------|--------------------------------------------------------------------------|
| Seq2Seq            | 0.528            | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task3/opennmt    |
| Pointer Generator	 | 0.561            | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task3/opennmt    |
| Transformer        | 0.528            | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task3/opennmt    |
| T5	                | 0.561            | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task3/t5         |
| ProphetNet	        | 0.561            | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task3/prophetnet |
