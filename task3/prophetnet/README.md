## Task 4：Medical Report Generation (MRG)

This dir contains the code of **ProphetNet** model for MRG task. 

The code is adapted from https://github.com/microsoft/ProphetNet/tree/master/ProphetNet_Dialog_Zh. 

The details are described in [ProphetNet-X paper](https://arxiv.org/abs/2104.08006).

### Requirements

- torch==1.3.0  
- fairseq==v0.9.0  
- tensorboardX==1.7    

### Preprocess 

```shell
python preprocess.py
```

### Binarization

```shell
fairseq-preprocess \
--user-dir prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--trainpref data/tokenized_train --validpref data/tokenized_dev --testpref data/tokenized_test \
--destdir processed --srcdict vocab.txt --tgtdict vocab.txt \
--workers 20
```

### Train

```shell
DATA_DIR=processed/
ARCH=ngram_transformer_prophet_large
CRITERION=ngram_language_loss
SAVE_DIR=models/
USER_DIR=./prophetnet
PRETRAINED_CHECKPOINT=./pretrained_checkpoint/prophetnet_zh.pt

fairseq-train $DATA_DIR \
--ngram 2 \
--user-dir $USER_DIR --task translation_prophetnet --arch $ARCH \
--optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
--lr 0.0001 --min-lr 1e-09 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--criterion $CRITERION --label-smoothing 0.1 \
--update-freq 4 --max-sentences 1 \
--num-workers 0  \
--ddp-backend=no_c10d --max-epoch 10 \
--max-source-positions 512 --max-target-positions 256 \
--truncate-source --load-from-pretrained-model $PRETRAINED_CHECKPOINT \
--empty-cache-freq 64 \
--save-dir $SAVE_DIR \
--distributed-no-spawn \
--skip-invalid-size-inputs-valid-test
```

### Inference

```shell
BEAM=5
CHECK_POINT=./models/checkpoint10.pt
TEMP_FILE=temp.txt
OUTPUT_FILE=test_pred.txt

fairseq-generate processed --path $CHECK_POINT --user-dir prophetnet --task translation_prophetnet --batch-size 80 --gen-subset test --beam $BEAM --num-workers 4 --no-repeat-ngram-size 3  2>&1 > $TEMP_FILE
grep ^H $TEMP_FILE | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > $OUTPUT_FILE
```

### Postprocess

```shell
cd .. || exit
python postprocess.py --gold_path ../../dataset/test_input.json --pred_path prophetnet/test_pred.txt --target no_t5
```

### Evaluation

```shell
python eval_task3.py --gold_path ../../dataset/test.json --pred_path prophetnet/test_pred.json
```
