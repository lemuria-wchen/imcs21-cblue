# Task1：BERT-NER

**医疗命名实体识别（IMCS-NER）** 任务的 *BERT-CRF* 基线模型。

## 0. Set Up

### 0.1 Dataset

下载数据集，将dataset文件夹放在Track1文件夹内。

### 0.2 Requirements

- python>=3.5
- torch>=1.4.0
- transformers==2.7.0
- seqeval==1.2.2
- pytorch-crf==0.7.2
- tqdm==4.42.1
- pandas>=1.0.3

## 1. Data Preprocess 

预处理训练数据，将生成ner_data文件夹

```
cd data
python data_preprocess.py
```

## 2. Training

```
python main.py --task ner_data --model_type bert --model_dir save_model --do_train --do_eval --use_crf
```

数据参数说明详见main.py的args。

## 3. Predicting

```
python predict.py --test_input_file {test_file_path} --test_output_file {output_file_path} --model_dir {saved_model_path}
```

`test_file_path`是`test.json`的路径，`output_file_path`是输出预测结果文件`submission_track1_task1.json`的路径，`saved_model_path`是保存的模型路径。

## 4. Evaluation

```
python eval_track1_task1.py {gold_data_path} {pred_data_path}
```

`gold_data_path`是具有真实标签的测试集的路径，`pred_data_path`是`submission_track1_task1.json`的路径。具体提交文件的命名及格式要求请参阅官网的结果提交示例。


## Experimental details

| Metric               | Value                   |
|----------------------|-------------------------|
| F1 score on test set | 92.21%                  |
| Training Epochs      | 10                      |
| Training Time        | 3.5h                    |
| CUDA                 | 10.1.243                |
| GPU                  | GeForce RTX 2080Ti 11GB |
| Linux Release        | Ubuntu 16.04.5 LTS      |


## References

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)
