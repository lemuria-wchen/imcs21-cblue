## IMCS-MRG Task

本文档包含 **IMCS-MRG** 任务的：

- 任务定义
- 提交格式
- 评价方法
- 基线模型

### 任务定义

**IMCS-MGR** 任务的目标是从医患对话中 *自动生成对应的诊疗报告*，该任务的输入输出示例如下。

| Input（自述 + 对话）                                                                                                           | Output（医疗报告）                                                                                                       |
|--------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| 【自述】<br>宝宝九个月了，嗓子有痰咳不出，很少咳嗽，怎么办<br><br>【对话】<br>...<br>医生：有没有发热<br>患者：没有<br>...<br>医生：用过什么药物<br>患者：给喝过小儿咳喘灵，阿莫西林颗粒<br>... | 主诉：有痰鸣音两天<br>现病史：患儿两天前咳嗽服药好转后，出现痰鸣音，<br>口服小儿咳喘灵，阿莫西林颗粒治疗，症状改善不明显<br>辅助检查：听诊<br>既往史：暂无<br>诊断：小儿支气管炎<br>建议：完善胸片，对症治疗 |


### 提交格式

**IMCS-MGR** 任务要求提交文件为 **json** 格式，具体格式如下。

```
{
    "example_id1": {                    # 测试集样本id
        "主诉": 
        "现病史":  
        "辅助检查":   
        "既往史": 
        "诊断": 
        "建议": 
    },
    "example_id2": {
        ...
    }, 
    ...
}
```

### 评价方法

**IMCS-MGR** 任务采用平均的 **Rouge** 得分 作为评价指标，详细见文件 `eval_task3.py`, 运行方式如下：

```shell
python3 eval_task3.py --gold_path {gold_file_path} --pred_path {pred_file_path}
```

### 基线模型

我们为 **IMCS-MGR** 任务创建了 5 个基线模型，模型详情如下。

| Model              | Avg. Rouge-score | Link                                                                     |
|--------------------|------------------|--------------------------------------------------------------------------|
| Seq2Seq            | 0.4797           | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task3/opennmt    |
| Pointer Generator	 | 0.5144           | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task3/opennmt    |
| Transformer        | 0.4772           | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task3/opennmt    |
| T5	                | 0.5426           | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task3/t5         |
| ProphetNet	        | 0.5421           | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task3/prophetnet |

