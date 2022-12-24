## IMCS21-CBLUE

这是 [**CBLUE@Tianchi**](https://tianchi.aliyun.com/specials/promotion/2021chinesemedicalnlpleaderboardchallenge) 中医疗对话数据集 **IMCS21** 的仓库，本仓库包含：

- 背景介绍
- CBLUE 评测任务介绍
- 基线模型代码
- 数据集介绍
- 多级数据注释
- 数据格式

### 更新

- IMCS21 数据集已更正部分标签，包括命名实体、症状标签等，并新增了`local_implicit_info`一个字段。
- IMCS21 数据集已更新，添加了 4 种疾病，覆盖了 10 种️疾病，共 4,116 条样本。新版本评测仓库更新了每个任务的评价脚本，同时也更新了基线代码。欢迎大家来 CBLUE 打榜！ 

**注意**：对 IMCS21 新版数据集的详细介绍请参考我们发表在生物信息领域期刊Bioinformatics 2022上的论文 [A Benchmark for Automatic Medical Consultation System: Frameworks, Tasks and Datasets](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btac817/6947983)，以及对应的代码仓库 [https://github.com/lemuria-wchen/imcs21](https://github.com/lemuria-wchen/imcs21)。 

### 背景介绍

随着"互联网+医疗"的迅速发展，在线问诊平台逐渐兴起，在线问诊是指医生通过对话和患者进行病情的交流、 疾病的诊断并且提供相关的医疗建议。在政策和疫情的影响之下，在线问诊需求增长迅速。然而医生资源是稀缺的，由此促使了**自动化医疗问诊**的发展，以人机对话来辅助问诊过程。为了促进智能医疗咨询系统（Intelligent Medical Consultation System, IMCS），我们构建了 **IMCS21** 数据集，该数据集收集了真实的在线医患对话，并进行了多层次（Multi-Level）的人工标注，包含**命名实体**、**对话意图**、**症状标签**、**医疗报告**等，我们将该数据集接入 **[CBLUE](https://tianchi.aliyun.com/specials/promotion/2021chinesemedicalnlpleaderboardchallenge)** 评测平台，希望可以共同促进智能医疗、医学语言理解等领域的发展。

### CBLUE 评测任务

**IMCS21** 目前在 [CBLUE](https://tianchi.aliyun.com/specials/promotion/2021chinesemedicalnlpleaderboardchallenge) 评测平台上接入了**四个任务**，分别为：命名实体识别、症状识别、医疗报告生成和对话意图识别。

| 任务编号 | 任务名称   | 简称       | 任务描述                                 | 链接                                                            |
|------|--------|----------|--------------------------------------|---------------------------------------------------------------|
| 任务一  | 命名实体识别 | IMCS-NER | 从医患对话文本中识别出五类重要的医疗相关实体。              | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task1 |
| 任务二  | 症状识别   | IMCS-SR  | 根据医患对话文本，识别出病人具有的症状信息（包含归一化标签和类别标签）。 | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task2 |
| 任务三  | 医疗报告生成 | IMCS-MRG | 依据病人自述和医患对话，输出具有规定格式的医疗报告。           | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task3 |
| 任务四  | 对话意图识别 | IMCS-DAC | 从话语中识别医生或者患者的意图（共 16 种）。             | https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task4 |

### 数据集介绍

**IMCS21** 包含 **4,116** 组细粒度标注的医患对话案例样本，覆盖 **10** 种儿科疾病，详细统计数据如下表所示。

| 统计指标                   | Statistical Metrics                                 | Value   |
|------------------------|-----------------------------------------------------|---------|
| 总疾病数                   | # of Diseases                                       | 10      |
| 总对话数                   | # of Total Dialogs                                  | 4,116   |
| 总句子数                   | # of Total Sentences                                | 164,731 |
| 平均每个对话包含的句子数           | # of Avg. Sentences / Per Dialog                    | 40      |
| 平均每个对话包含的字符数           | # of Avg. Words / Per Dialog                        | 523     |
| 平均每个对话包含的字符数（包含患者自我报告） | # of Avg. Words / Per Dialog (Contains Self-Report) | 580     |

| 疾病名称    | 样本数 |
|---------|-----|
| 小儿支气管炎  | 543 |
| 小儿发热    | 542 |
| 小儿腹泻    | 534 |
| 上呼吸道感染  | 486 |
| 小儿消化不良  | 475 |
| 小儿感冒    | 472 |
| 小儿咳嗽    | 344 |
| 新生儿黄疸   | 294 |
| 小儿便秘    | 221 |
| 小儿支气管肺炎 | 205 |

### 多级数据注释

#### 命名实体

医疗命名实体广泛存在于医患对话中，它们是理解对话意图的关键因素，也是构建智能医疗对话系统的基础任务。

**IMCS21** 中标注了对话中医疗相关的命名实体，共包含 **5** 类命名实体，标注方式采用 *BIO* 三位 **字符级标注** ，其中 *B-X* 代表实体 *X* 的开头，*I-X* 代表实体的结尾，*O* 代表不属于任何类型，命名实体的预定义类别定义如下。

| 实体类别 | Entity Category     | 类别详情                               |
|------|---------------------|------------------------------------|
| 症状   | Symptom             | 病人因患病而表现出来的异常状况，如 *发热、呼吸困难、鼻塞* 等。  |
| 药品名  | Drug                | 具体的药物名称，如 *妈咪爱、蒙脱石散、蒲地蓝* 等。        |
| 药物类别 | Drug_Category       | 根据药物功能进行划分的药物种类，如 *消炎药、感冒药、益生菌* 等。 |
| 检查   | Medical_Examination | 医学检验，如 *血常规、x光片、CRP分析* 等。          |
| 操作   | Operation           | 相关的医疗操作，如 *输液、雾化、接种疫苗* 等。          |

#### 对话意图

**IMCS21** 中标注了医患对话行为，共包含 **16** 类对话意图，标注方式采用 **句子级** 标注，对话意图的预定义类别定义如下。

| 对话意图类别     | Dialogue Intent Category                   | Dominant |
|------------|--------------------------------------------|----------|
| 提问-症状      | Request-Symptom                            | 医生       |
| 告知-症状      | Inform-Symptom                             | 病人       |
| 提问-病因      | Request-Etiology                           | 医生       |
| 告知-病因      | Inform-Etiology                            | 病人       |
| 提问-基本信息    | Request-Basic_Information                  | 医生       |
| 告知-基本信息    | Inform-Basic_Information                   | 病人       |
| 提问-已有检查和治疗 | Request-Existing_Examination_and_Treatment | 医生       |
| 告知-已有检查和治疗 | Inform-Existing_Examination_and_Treatment  | 病人       |
| 提问-用药建议    | Request-Drug_Recommendation                | 病人       |
| 告知-用药建议    | Inform-Drug_Recommendation                 | 医生       |
| 提问-就医建议    | Request-Medical_Advice                     | 病人       |
| 告知-就医建议    | Inform-Medical_Advice                      | 医生       |
| 提问-注意事项    | Request-Precautions                        | 病人       |
| 告知-注意事项    | Inform-Precautions                         | 医生       |
| 诊断         | Diagnose                                   | 医生       |
| 其他         | Other                                      | 医生 / 病人  |

#### 症状标签

症状是医患对话中主要讨论的话题之一，病人的症状信息也是对话策略和疾病诊断的关键特征。使用 *BIO* 标签可以找出症状实体所在的位置，然而在实际应用中，还存在两个问题：1) 症状实体未归一化，相同的症状可能有多种表达方式，如 *发烧、热、发热* 都表示 *发热* 这一症状；2) 症状与患者的关系未知，显然病人并非一定患有所有出现在对话中的症状。

**IMCS21** 对症状实体进行了进一步的 **实体级** 标注，我们标注了每个症状实体的 **归一化标签** 和 **类别标签**，这两种标签的详情如下。

| 症状标签  | Symptom Labels | Details                                                     |
|-------|----------------|-------------------------------------------------------------|
| 归一化标签 | symptom_norm   | 从 *BIO* 标签中提取的 **1,900** 多个症状中标准化得到 **444** 个标准化症状名称。       |
| 类别标签  | symptom_type   | "0" 代表确定病人没有患有该症状，"1" 代表确定病人患有该症状，"2" 代表无法根据上下文确定病人是否患有该症状。 |


#### 医疗报告

医疗报告是医生对病人健康状况的总结，是医疗诊断过程的重要环节。

**IMCS21** 标注了医疗报告。标注方式采用 **对话级标注**，标注者阅读完整医患对话，并按照规定格式为患者填写对应的医疗报告，每个对话均包含 **2** 份医疗报告作为参考。医疗报告的预定义格式如下。

| 字段   | Fields          | Details               |
|------|-----------------|-----------------------|
| 主诉   | Chief Complaint | 病人自诉（Self-report）的总结。 |
| 现病史  | Present Disease | 对话中病人涉及到的现病史的总结。      |
| 辅助检查 | Auxiliary       | 对话中病人涉及过的医疗检查的总结。     |
| 即往史  | Past History    | 对话中医生对病人的过去病史的总结。     |
| 诊断   | Diagnosis       | 对话中医生对病人的诊断结果的总结。     |
| 建议   | Suggestions     | 对话中医生对病人的建议的总结。       |


### 数据格式

#### 训练集 

文件名为为 `train.json`，共 **2,472** 条样本，其格式如下。

```
{
  "example_id1":{	                # 样本id
      "diagnosis":	                # 患者疾病类别
      "self-report":	            # 自诉，病人对自己病情的陈述及对医生的提问
      "dialogue":[	                # 对话内容
        {
          "sentence_id":	        # 对话轮次的序号
          "speaker":		        # 医生或者患者
          "sentence":		        # 当前对话文本内容
          "dialogue_act":	        # 话语行为
          "BIO_label":	            # BIO实体标签（以“空格”连接）（第二版已修正部分错误标签）
          "symptom_norm":	        # 归一化的症状（与BIO中的症状出现的顺序对应）（第二版已修正部分错误标签）
          "symptom_type":	        # 症状类别（与BIO中的症状出现的顺序对应）（第二版已修正部分错误标签）
          "local_implicit_info":    # 每句话中包含的"症状-标签"字典（由 symptom_norm 和 symptom_type 构建，规则如下，如果一个句子中出现相同的两个 symptom_norm，则 symptom_type 的优先级为：'1' > '0' > '2'，即，如果在同一个句子中，相同的 symptom_norm 有多个不同的 symptom_type，我们仅按照优先级保留其中的一个 symptom_type，这也是用户需要预测的结果）    
        },
        {	
          "sentence_id":
          "speaker":
          "sentence":
          "dialogue_act":
          "BIO_label":
          "symptom_norm":	
          "symptom_type":
          "local_implicit_info":  
        },
        ...
      ]
      "report":[                   # 两份医疗报告
      {
        "主诉": 
        "现病史":
        "辅助检查": 
        "既往史": 
        "诊断": 
        "建议": 
      }, 
      {
        "主诉": 
        "现病史":  
        "辅助检查": 
        "既往史": 
        "诊断": 
        "建议": 
      },       
      ]     
      "explicit_info":{
          "Symptom": 	            # 患者自我报告中的症状，列表格式，值为症状的归一化标签
      }         
      "implicit_info":{
          "Symptom": 	            # 基于整组对话推断得到的症状标签，字典格式，键为症状的归一化标签，值为症状的类别标签（第二版已修正部分错误标签）
      } 
  }
  "example_id2":{
      ...
  }
  ...
}
```

#### 验证集

文件名为为 `dev.json`，共 **833** 条样本，其格式同训练集。

#### 测试集输入（仅包含原始医患对话）

文件名为为 `test_input.json`，共 **811** 条样本，其格式如下。

```
{
  "example_id1":{	                # 样本id
      "self-report":	            # 自诉，病人对自己病情的陈述及对医生的提问
      "dialogue":[	                # 对话内容
        {
          "sentence_id":	        # 对话轮次的序号
          "speaker":		        # 医生或者患者
          "sentence":		        # 当前对话文本内容
        },
        {	
          "sentence_id":
          "speaker":
          "sentence":
        },
        ...
      ]
  }
  "example_id2":{
      ...
  }
  ...
}
```

#### 验证集完整

文件名为为 `test.json`，共 **811** 条样本，其格式同训练集与验证集。

#### 归一化的症状词典（optional，第一版）

文件名为为 `symptom_norm.csv`，归一化后的症状词典，**本仓库部分基线代码依赖该文件**，其格式如下。

```
norm
咳嗽
发热
感冒
...
```

#### 映射字典（optional，第二版）

文件名为为 `symptom_norm.csv`，归一化后的症状词典，**本仓库部分基线代码依赖该文件**，其格式如下。

```
norm
咳嗽
发热
感冒
...
```

### 引用

如果您扩展或使用这项工作，请引用介绍它的[论文](https://arxiv.org/abs/2204.08997)。

```
@article{chen2022benchmark,
  title={A Benchmark for Automatic Medical Consultation System: Frameworks, Tasks and Datasets},
  author={Chen, Wei and Li, Zhiwei and Fang, Hongyi and Yao, Qianyuan and Zhong, Cheng and Hao, Jianye and Zhang, Qi and Huang, Xuanjing and Wei, Zhongyu and others},
  journal={arXiv preprint arXiv:2204.08997},
  year={2022}
}
```
