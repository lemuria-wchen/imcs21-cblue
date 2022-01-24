# Requirement

python3.6 ，cuda10.0

torch==1.2.0

tensorflow==1.13.1

pandas==0.25.1

sumeval==0.2.2

其他依赖包和版本可以参考 requirement.txt

# 数据

* 数据说明

数据从[官网下载](http://210.16.187.147:20023/download)，并保存至dataset文件夹中。

训练集和验证集参赛选手可以重新进行划分。但是测试集是固定的。

参赛选手可以选择使用数据集中包含的其他信息（如：实体信息）改进模型，但是要求测试时的输入为主诉和对话。（比赛第二阶段会检查模型）

* 数据预处理

```
python make_datafiles.py 
```

处理好后的文件会存在medi_finished_dir 文件夹中，其中vocab是基于训练集中获取的字典。file_names_train，file_names_dev和file_names_test保存了example id。

# seq2seq模型

- 训练

 ```python
python train.py --use_gpu --exp_name=s2s 
 ```

参数说明见train.py的args，可以使用python train -h 进行查看。

* 验证

```python
python decode.py --model_filename=<model_dir> --decode_filename=medi_finished_dir/dev.bin --mode=dev --compute_rouge  --output_filenames=medi_finished_dir/file_names_dev
```

<model_dir> 中填写模型的地址

* 测试

```python
python decode.py --model_filename=<model_dir>
```

生成用于测试的文件。将“生成摘要_test”文件夹上传，作为提交评测的文件。具体提交文件的命名及格式要求请参阅官网的结果提交示例。


# pointer_generator模型

* 训练

```python
python train.py --use_gpu --pointer_gen --is_coverage --exp_name=pg
```

* 验证

``` python
python decode.py --model_filename=<model_dir> --decode_filename=medi_finished_dir/dev.bin --mode=dev --compute_rouge  --output_filenames=medi_finished_dir/file_names_dev --pointer_gen --is_coverage 
```

* 测试

```python
python decode.py --model_filename=<model_dir> --pointer_gen --is_coverage
```
生成用于测试的文件。将“生成摘要_test”文件夹上传，作为提交评测的文件。具体提交文件的命名及格式要求请参阅官网的结果提交示例。


# 第一阶段评测说明

在文件夹[evaluation](https://github.com/Guardianzc/MedicalChatBox/edit/main/track2/evaluation)中，给出了第一阶段提交的文件样例和自动化评价的代码。

- evaluation 文件夹中的vocab是包含出现在数据集中的所有字。

- 自动化测评的代码：evaluate.py。其中，使用[sumeval](https://github.com/chakki-works/sumeval)计算ROUGE值。

- 提交文件要求：命名为“生成摘要_test" 的文件夹中含有以“test_XXXX"命名的文件，XXXX是example_id。每个文件是生成的诊疗报告。其中每个字符用空格‘ ’隔开。


- Baseline 自动化评测得分：

| Model             | ROUGE-1 | ROUGE-2 | ROUGE-L |
| :---------------- | :-----: | :-----: | :-----: |
| seq2seq           | 0.5850  | 0.4346  | 0.5639  |
| pointer-generator | 0.6213 | 0.4700  | 0.5905  |

# References
- [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
- [Task-oriented Dialogue System for Automatic Diagnosis](https://www.aclweb.org/anthology/P18-2033.pdf)
- [Github - abisee/pointer_summarizer](https://github.com/abisee/pointer-generator)


