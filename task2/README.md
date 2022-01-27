## IMCS-SR Task

本文档包含 **症状识别（IMCS-SR）** 任务的：

- 任务定义
- 提交格式
- 评价方法
- 基线模型
- 参考文献

### 任务定义

**IMCS-SR** 任务的目标是从医患对话中 *同时预测症状的归一化标签和类别标签*，该任务的输入输出示例如下。

| Input（对话）                                          | Outputs （症状及其属性）      |
|----------------------------------------------------|-----------------------|
| ...<br>患者：没有**发热**，但是**咳嗽**<br>...<br>患者：嗓子里有*呼噜声* | 发热：0<br>咳嗽：1<br>痰鸣音：1 |


### 提交格式

**IMCS-SR** 任务要求提交文件为 **json** 格式，具体格式如下。

```
{
    "example_id1": { # 测试集样本id
        "咳嗽": "1",
        "发热": "0",
        "鼻流涕": "2"
    }
   	"example_id2":{
   	...
   	}
...
}
```

### 评价方法

**IMCS-SR** 任务采用 **F1 score** 作为评价指标，详细见文件 `eval_task2.py`, 运行方式如下：

```shell
python3 eval_task2.py {gold_data_path} {pred_data_path}
```

### 基线模型

我们为 **IMCS-SR** 任务创建了 2 个基线模型，模型详情如下。

| Model | F1-score (%) | Link                                                              |
|-------|--------------|-------------------------------------------------------------------|
| MLC   | 67.99        | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task2/MLC |
| MTL   | 69.86        | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task2/MTL |

### 参考文献

```markdown
@inproceedings{2019Enhancing,
  title={Enhancing Dialogue Symptom Diagnosis with Global Attention and Symptom Graph},
  author={X Lin and X He and Chen, Q. and Tou, H. and Chen, T.},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  year={2019},
}
```
