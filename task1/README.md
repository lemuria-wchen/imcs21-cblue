# 赛道一：医患对话理解

赛道一共包含两个任务，分别是命名实体识别任务以及症状识别任务。

本文件夹包括：（1）数据集dataset；（2）评价文件`eval_track1_task1.py`、`eval_track1_task2.py`

## 1. 数据集

赛道一使用的数据集存放在`./dataset/`文件夹中，共包含以下2个文件：

* **train.json**，**比第一阶段新增两种疾病共810条样本**

  训练和验证所使用的数据集，其格式如下：

  ```
  {
  	"example_id1"：{	#样本id
  		"diagnosis":	#患者疾病类别
  		"self-report":	#自诉，病人对自己病情的陈述及对医生的提问
  		"dialogue":[	#对话内容
  				{	"sentence_id":	#对话轮次的序号
  					"speaker":		#医生或者患者
  					"sentence":		#当前对话文本内容
  					"dialogue_act":	#话语行为
  					"BIO_label":	#BIO实体标签（以“空格”连接）
  					"symptom_norm":	#归一化的症状（与BIO中的症状出现的顺序对应）
  					"symptom_type":	#症状类别（与BIO中的症状出现的顺序对应）
  				},
  				{	"sentence_id":
  					"speaker":
  					"sentence":
  					"dialogue_act":
  					"BIO_label":
  					"symptom_norm":	
  					"symptom_type":
  				},
  				…
  		]
  		"report":		#医疗报告[report1, report2]
  		"implicit_info":{
  			"Symptom": 	#整组对话归一化后症状及类别标签，key是归一化症状、value是症状0/1/2类别
  		}
  	}
  	"example_id2":{
  	…
  	}
  	…
  }
  ```

  其中标注内容的详细说明如下：

  - BIO_label: 对话中的BIO实体信息，一共5类：症状、药品名、药物类别、检查和操作。
    中英文对应关系如下：{'症状':'Symptom','药品名':'Drug','药物类别':'Drug_Category','检查':'Medical_Examination','操作':'Operation'}
  - dialogue_act: 对话中的话语行为，一共16类：提问-症状，告知-症状，提问-病因，告知-病因，提问-基本信息，告知-基本信息，提问-已有检查和治疗，告知-已有检查和治疗，提问-用药建议，告知-用药建议，提问-就医建议，告知-就医建议，提问-注意事项，告知-注意事项，诊断，其他。
    中英文对应关系如下：{'提问-症状':'Request-Symptom','告知-症状':'Inform-Symptom','提问-病因':'Request-Etiology','告知-病因':'Inform-Etiology','提问-基本信息':'Request-Basic_Information','告知-基本信息':'Inform-Basic_Information','提问-已有检查和治疗':'Request-Existing_Examination_and_Treatment', '告知-已有检查和治疗':'Inform-Existing_Examination_and_Treatment','提问-用药建议':'Request-Drug_Recommendation','告知-用药建议':'Inform-Drug_Recommendation','提问-就医建议':'Request-Medical_Advice','告知-就医建议':'Inform-Medical_Advice', '提问-注意事项':'Request-Precautions','告知-注意事项':'Inform-Precautions','诊断':'Diagnose','其他':'Other'}
  - symptom_norm：症状实体的归一化标签，与BIO中的症状实体顺序对应。
  - symptom_type:  症状实体的类别标签，是结合整段对话得到的标签，表示患者是否已经有该症状。0代表没有，1代表有，2代表不确定。
  - report: 诊疗报告，每个样本均有两个诊疗报告。包含六部分：(1) 主诉： 主要症状或体征 (2) 现病史： 主要症状的描述（发病情况，发病时间） (3) 辅助检查：病人已有的检查项目、检查结果、会诊记录等 (4) 既往史：既往的健康状况、过去曾经患过的疾病等 (5) 诊断：对疾病的诊断 (6) 建议：检查建议、药物治疗、注意事项。
  - implicit_info: 病人整组对话中提及的症状和检查信息，以及它们的类别标签。

* **symptom_norm.csv**

  共包含329个归一化后的症状实体。

## 2. 测试结果文件

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

* 任务二（症状识别）模型预留接口的输出结果文件需为json格式，文件的格式如下：

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



## 3. 评价方式

* 任务一（命名实体识别）采用**F1值**作为评价指标，详细见文件`eval_track1_task1.py`, 运行方式如下：

```
python eval_track1_task1.py {gold_data_path} {pred_data_path}
```

* 任务二（症状识别）采用**F1值**作为评价指标，详细见文件eval_track1_task2.py，运行方式如下：

```
python eval_track1_task2.py {gold_data_path} {pred_data_path}
```

**赛道一最终的评价方式为：任务一F1和任务二F1的平均**



## 4. 备注

```
import json
# json文件读入
with open(data_path, 'r', encoding='utf-8') as f:
	data = json.load(f)
# json文件写入
with open(data_path, 'w', encoding='utf-8') as f:
	json.dump(data, f, ensure_ascii=False, indent=4)
```



## 参考

```markdown
@inproceedings{2019Enhancing,
  title={Enhancing Dialogue Symptom Diagnosis with Global Attention and Symptom Graph},
  author={X Lin and X He and Chen, Q. and Tou, H. and Chen, T.},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  year={2019},
}
```



