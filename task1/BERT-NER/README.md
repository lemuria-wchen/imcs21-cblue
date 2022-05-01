## Medical Named Entity Recognition (Medical NER)

This dir contains the code of **BERT-CRF** model for the Medical NER task. 

It also supports **RoBerta** and **MacBert**.

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
python main.py --task ner_data --model_type bert --model_dir saved/bert --do_train --do_eval --use_crf
```

### Inference

```shell
python predict.py --test_input_file ../../../dataset/test_input.json --test_output_file pred_bert.json --model_dir saved/bert
```

### Evaluation

```shell
cd .. || exit
python eval_task1.py --gold_path ../../dataset/test.json --pred_path BERT-NER/pred_bert.json
```
