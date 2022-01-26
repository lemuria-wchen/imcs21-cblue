## 任务一: 医疗命名实体识别（IMCS-NER）

本任务是从医患对话文本中识别出重要的实体，共有五类医疗相关实体，采用字符级别的"BIO"标注体系进行标注。

### 任务

| Input（对话）                  | Output （BIO标签）                                                                             |
|----------------------------|:-------------------------------------------------------------------------------------------|
| ...                        | ...                                                                                        |
| 医生：有没有**发热**               | O O O O O O O B-Symptom I-Symptom                                                          |
| 患者：没有                      | O O O O O                                                                                  |
| ...                        | ...                                                                                        |
| 医生：用过什么药物                  | O O O O O O O O O                                                                          |
| 患者：给喝过**小儿咳喘灵**，**阿莫西林颗粒** | O O O O O O B-Drug I-Drug I-Drug I-Drug I-Drug O B-Drug I-Drug I-Drug I-Drug I-Drug I-Drug |
| ...                        | ...                                                                                        |


### 提交

* 任务一（命名实体识别）模型预留接口的输出结果文件需为json格式，文件的格式如下：

```
{
    "example_id1": { # 测试集样本id
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

## 3. 评价方式

* 任务一（命名实体识别）采用**F1值**作为评价指标，详细见文件`eval_track1_task1.py`, 运行方式如下：

```
python eval_track1_task1.py {gold_data_path} {pred_data_path}
```



### 备注

```python
import json
data_path = ''
# json文件读入
with open(data_path, 'r', encoding='utf-8') as f:
	data = json.load(f)
# json文件写入
with open(data_path, 'w', encoding='utf-8') as f:
	json.dump(data, f, ensure_ascii=False, indent=4)
```


### 参考

```markdown
@inproceedings{2019Enhancing,
  title={Enhancing Dialogue Symptom Diagnosis with Global Attention and Symptom Graph},
  author={X Lin and X He and Chen, Q. and Tou, H. and Chen, T.},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  year={2019},
}
```
