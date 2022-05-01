## Task 4：Medical Report Generation (MRG)

This dir contains the code of **LSTM / Pointer-Generator / Transformer** model for MRG task. 

We use [OpenNMT](https://github.com/OpenNMT/OpenNMT-py), an open-source neural machine translation toolkit. 

The input is the concatenation of all non-OTHER categories of utterances, and the output is a flat medical report that expands and concatenate the key and value. 

During inference, the utterance of category is pre-predicted by ERNIE model trained in DAC task.

### Requirements

- PyTorch==1.6.0
- Python >= 3.6

```shell
pip install OpenNMT-py
```

### Preprocess

```shell
python preprocess.py
```

#### LSTM

```shell
# build the vocab
onmt_build_vocab -config lstm.yaml -share_vocab -n_sample 10000

# train
onmt_train -config lstm.yaml

# inference
onmt_translate -model saved/lstm_step_20000.pt -src data/src-test.txt -output data/pred_lstm.txt -gpu 0
```

#### Pointer-Generator

```shell
# build the vocab
onmt_build_vocab -config pg.yaml -share_vocab -n_sample 10000

# train
onmt_train -config pg.yaml

# inference
onmt_translate -model saved/pg_step_20000.pt -src data/src-test.txt -output data/pred_pg.txt -gpu 0
```

#### Transformer

```shell
# build the vocab
onmt_build_vocab -config transformer.yaml -share_vocab -n_sample 10000

# train
onmt_train -config transformer.yaml

# inference
onmt_translate -model saved/tf_step_20000.pt -src data/src-test.txt -output data/pred_tf.txt -gpu 0
```

### Postprocess

```shell
cd .. || exit
python postprocess.py --gold_path ../../dataset/test_input.json --pred_path opennmt/data/pred_lstm.txt --target no_t5
python postprocess.py --gold_path ../../dataset/test_input.json --pred_path opennmt/data/pred_pg.txt --target no_t5
python postprocess.py --gold_path ../../dataset/test_input.json --pred_path opennmt/data/pred_tf.txt --target no_t5
```

### Evaluation

```shell
python eval_task3.py --gold_path ../../dataset/test.json --pred_path opennmt/data/pred_lstm.json
python eval_task3.py --gold_path ../../dataset/test.json --pred_path opennmt/data/pred_pg.json
python eval_task3.py --gold_path ../../dataset/test.json --pred_path opennmt/data/pred_tf.json
```
