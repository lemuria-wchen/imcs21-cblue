## Symptom Recognition (SR)

This dir contains the code of **BERT-MLC** model for SR task. 

### Requirements

- python>=3.7
- torch==1.8.1
- transformers==4.5.1
- pandas==1.2.0
- numpy==1.19.2
- sklearn

```shell
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Preprocess

```shell
python preprocess.py --target imp
```

### Training

```shell
python train.py --target imp --cuda_num 0
```

### Inference

```shell
python inference.py --dataset test --target imp --cuda_num 0
```

### Evaluation

```shell
cd .. || exit
python eval_task2.py --gold_path ../../dataset/test.json --pred_path MLC-SLI/test_imp_pred.json
```
