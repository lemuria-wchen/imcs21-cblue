## IMCS21-CBLUE

这是CBLUE@Tianchi中医疗对话数据集 **IMCS21** 的仓库，本仓库包含：

- 背景介绍
- 数据集详情
- 评测任务及基线模型代码
  - [任务一: 医疗命名实体识别（IMCS-NER）](https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task1)
  - [任务二：症状识别（IMCS-SR）](https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task2)
  - [任务三：医疗报告生成（IMCS-MRG）](https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task3)

### 背景

随着"互联网+医疗"的迅速发展，在线问诊平台逐渐兴起，在线问诊是指医生通过对话和患者进行病情的交流、 疾病的诊断并且提供相关的医疗建议。

在政策和疫情的影响之下，在线问诊需求增长迅速。然而医生资源是稀缺的，由此促使了**自动化医疗问诊**的发展，以人机对话来辅助问诊过程。

### 数据集详情

本数据集 **IMCS21** 包含 **3,052** 组细粒度标注的医患对话案例样本，覆盖 **6** 种儿科疾病，详细统计数据如下表所示。

| 统计指标         | Statistical Metrics              | Value   |
|--------------|----------------------------------|---------|
| 覆盖疾病数        | # of Diseases                    | 6       |
| 总对话数         | # of Total Dialogs               | 3,052   |
| 总句子数         | # of Total Sentences             | 123,762 |
| 平均每个对话包含的句子数 | # of Avg. Sentences / Per Dialog | 40.55   |
| 平均每个对话包含的字符数 | # of Avg. Words / Per Dialog     | 531.18  |


| 疾病名称   | 样本数 |
|--------|-----|
| 小儿支气管炎 | 553 |
| 小儿发热   | 542 |
| 小儿腹泻   | 534 |
| 上呼吸道感染 | 486 |
| 小儿消化不良 | 475 |
| 小儿感冒   | 472 |

### 多级数据注释











