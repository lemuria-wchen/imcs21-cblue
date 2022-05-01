## seq2seq (lstm)

# build the vocab
onmt_build_vocab -config lstm.yaml -share_vocab -n_sample 10000

# train
onmt_train -config lstm.yaml -share_vocab

# inference
onmt_translate -model saved/lstm_step_20000.pt -src data/src-test.txt -output data/pred_lstm.txt -gpu 0

## pointer generator

# build the vocab
onmt_build_vocab -config pg.yaml -share_vocab -n_sample 10000

# train
onmt_train -config pg.yaml -share_vocab

# inference
onmt_translate -model saved/pg_step_20000.pt -src data/src-test.txt -output data/pred_pg.txt -gpu 0

## transformer
# build the vocab
onmt_build_vocab -config transformer.yaml -share_vocab -n_sample 10000

# train
onmt_train -config transformer.yaml -share_vocab

# inference
onmt_translate -model saved/tf_step_20000.pt -src data/src-test.txt -output data/pred_tf.txt -gpu 0

# for dev set
onmt_translate -model saved/lstm_step_20000.pt -src data/src-dev.txt -output data/pred_lstm_dev.txt -gpu 0
onmt_translate -model saved/pg_step_20000.pt -src data/src-dev.txt -output data/pred_pg_dev.txt -gpu 0
onmt_translate -model saved/tf_step_20000.pt -src data/src-dev.txt -output data/pred_tf_dev.txt -gpu 0
