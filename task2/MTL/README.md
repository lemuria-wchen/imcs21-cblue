# Track1- Task2：MTL-NEAR

[第一届智能对话诊疗评测比赛imcs21](http://www.fudan-disc.com/sharedtask/imcs21/index.html)赛道一任务二（症状识别）的多任务学习基础模型。

## 0. Set Up

### 0.1 Dataset

数据集从网站[imcs21](http://www.fudan-disc.com/sharedtask/imcs21/index.html)中下载，将dataset文件夹放在Track1文件夹内。

### 0.2 Requirements

- python>=3.5
- tensorflow==1.13.1
- pandas>=1.0.3

## 1. Data Preprocess 

预处理训练数据，将生成near_data文件夹

```
cd data
python data_preprocess.py
```

## 2. Training

```
python train.py --data_dir data/near_data --save_dir save_model --do_train
```

模型参数说明见train.py的args。

## 3. Predicting

```
python train.py --test_input_file {test_file_path} --test_output_file {output_file_path} --save_dir {saved_model_path} --do_predict
```

`test_file_path`是`test.json`的路径，`output_file_path`是输出预测结果文件`submission_track1_task2.json`的路径，`saved_model_path`是保存的模型路径。

## 4. Evaluation

```
python eval_track1_task2.py {gold_data_path} {pred_data_path}
```

`gold_data_path`是具有真实标签的测试集的路径，`pred_data_path`是`submission_track1_task2.json`的路径。具体提交文件的命名及格式要求请参阅官网的结果提交示例。


## Attention

为训练方便，此模型在训练和验证时是根据每个句子识别出的症状及类别计算F1值。在测试中将同一个样本的所有句子识别出的症状及类别用投票规则进行统一，再计算F1值。实验结果显示效果差别不大，最终评价方式将以样本级别为准，参照`eval_track1_task2.py`。


## Experimental details

| Metric               | Value                   |
| -------------------- | ----------------------- |
| F1 score on test set | 69.86%                  |
| Training Epochs      | 10                      |
| Training Time        | 4h                      |
| CUDA                 | 10.1.243                |
| GPU                  | GeForce RTX 2080Ti 11GB |
| Linux Release        | Ubuntu 16.04.5 LTS      |
