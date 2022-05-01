## Symptom Recognition (SR)

This dir contains the code of **BERT-MTL** model for SR task. 

The model utilizes the BIO labels and contains three training objectives: 

- BIO tag prediction
- Symptom name classification
- Symptom label classification

The model is only valid for SLI-IMP task since it requires the annotated BIO tags. **DialoAMC do not provide BIO tags for self-reports**. 

### Requirements

- python>=3.5
- torch>=1.4.0
- transformers==2.7.0
- seqeval==1.2.2
- pytorch-crf==0.7.2
- tqdm==4.42.1
- pandas>=1.0.3

```shell
pip install -r requirements.txt
```

### Preprocess

```shell
python preprocess.py
```

### Training

```shell
python train.py --cuda_num 0
```

### Inference

```shell
python inference.py --dataset test --cuda_num 0 
```

### Evaluation

```shell
cd .. || exit
python eval_task2.py --gold_path ../../dataset/test.json --pred_path MTL-SLI/test_imp_pred.json
```
