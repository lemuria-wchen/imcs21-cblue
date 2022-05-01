## Task 4：Medical Report Generation (MRG)

This dir contains the code of **T5** model for MRG task. 

The code is adapted from https://github.com/SunnyGJing/t5-pegasus-chinese. 

### Requirements

- transformers==4.15.0  
- tokenizers==0.10.3  
- torch==1.7.0,1.8.0,1.8.1
- jieba
- rouge
- tqdm
- pandas 
- sympy

```shell
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Preprocess 

```shell
python preprocess.py
```

### Train

```shell
python train_with_finetune.py
```

### Inference

```shell
python predict_with_generate.py --test_data ./data/predict.tsv --result_file ./data/predict_result.tsv --use_multiprocess
```

### Postprocess

```shell
cd .. || exit
python postprocess.py --gold_path ../../dataset/test_input.json --pred_path t5/data/predict_result.tsv --target t5
```

### Evaluation

```shell
python eval_task3.py --gold_path ../../dataset/test.json --pred_path t5/data/predict_result.json
```
