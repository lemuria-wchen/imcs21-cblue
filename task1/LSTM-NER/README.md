## Medical Named Entity Recognition (Medical NER)

This dir contains the code of **LSTM-CRF** model for the Medical NER task.

### Requirements

- python==3.6
- tensorflow==1.13.1
- seqeval==1.2.2
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
python train.py --do_train
```

### Inference

```shell
python train.py --test_input_file ../../../dataset/test_input.json --test_output_file pred_lstm_test.json --save_dir saved/lstm --do_predict
```

### Evaluation

```shell
cd .. || exit 
python eval_ner.py --gold_path ../../dataset/test.json --pred_path LSTM-NER/pred_lstm_test.json
```
