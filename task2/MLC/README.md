# Track1- Task2：MLC-NEAR

[第一届智能对话诊疗评测比赛imcs21](http://www.fudan-disc.com/sharedtask/imcs21/index.html)赛道一任务二（症状识别）的多标签分类基础模型。

## 0. Set Up

### 0.1 Dataset

数据集从网站[imcs21](http://www.fudan-disc.com/sharedtask/imcs21/index.html)中下载，将dataset文件夹放在Track1文件夹内。

### 0.2 Requirements

- python>=3.7
- torch==1.8.1
- transformers==4.5.1
- pandas
- sklearn
- numpy

## 1. Data Preprocess 

预处理训练数据，将在data文件夹下生成processed文件夹

```
cd data
python preprocess.py
```

## 2. Training

```
python train.py
```

## 3. Predicting

```
python inference.py
```
将在data文件夹下输出预测结果文件`submission_track1_task2.json`


## 4. Evaluation

```
python eval_track1_task2.py {gold_data_path} {pred_data_path}
```

`gold_data_path`是具有真实标签的测试集的路径，`pred_data_path`是`submission_track1_task2.json`的路径。具体提交文件的命名及格式要求请参阅官网的结果提交示例。


## Experimental details

| Metric               | Value                |
| -------------------- | -------------------- |
| F1 score on test set | 67.9919%             |
| Training Epochs      | 20                   |
| Training Time        | 8h                   |
| CUDA                 | 10.1.243             |
| GPU                  | Tesla P100 PCIe 16GB |
| Linux Release        | Ubuntu 18.04.5 LTS   |

| Package              | Version              |
| -------------------- | -------------------- |
| Python               | 3.7                  |
| torch                | 1.8.1                |
| transformers         | 4.5.1                |
| pandas               | 1.2.0                |
| numpy                | 1.19.2               |
