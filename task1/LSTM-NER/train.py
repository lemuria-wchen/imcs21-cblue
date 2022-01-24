import logging
import tensorflow as tf
import os
import argparse
import random
import json

from model import MyModel
from utils import DataProcessor_LSTM as DataProcessor
from utils import DataProcessor_LSTM_Test as DataProcessor_Test
from utils import load_vocabulary
from utils import compute_metrics

logger = logging.getLogger()


def init_logging(args):
    """logging设置和参数信息打印"""
    log_file_path = os.path.join(args.save_dir, "run.log")
    if os.path.exists(log_file_path) and args.do_train == True:
        os.remove(log_file_path)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    fhlr = logging.FileHandler(log_file_path)
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)

    logger.info("====== parameters setting =======")
    logger.info("data_dir: " + str(args.data_dir))
    logger.info("save_dir: " + str(args.save_dir))
    logger.info("test_input_file: " + str(args.test_input_file))
    logger.info("test_output_file: " + str(args.test_output_file))
    logger.info("do_train: " + str(args.do_train))
    logger.info("do_predict: " + str(args.do_predict))
    logger.info("use_crf: " + str(args.use_crf))
    logger.info("word_embedding_dim: " + str(args.word_embedding_dim))
    logger.info("encoder_hidden_dim: " + str(args.encoder_hidden_dim))
    logger.info("num_epoch: " + str(args.num_epoch))
    logger.info("batch_size: " + str(args.batch_size))
    logger.info("random_seed: " + str(args.random_seed))
    logger.info("evaluate_steps: " + str(args.evaluate_steps))

def get_vocab(args):
    """获得字典"""
    logger.info("loading vocab...")
    w2i_char, i2w_char = load_vocabulary(os.path.join(args.data_dir, "vocab_char.txt"))
    w2i_bio, i2w_bio = load_vocabulary(os.path.join(args.data_dir, "vocab_bio.txt"))
    vocab_dict = {
        "w2i_char": w2i_char,
        "i2w_char": i2w_char,
        "w2i_bio": w2i_bio,
        "i2w_bio": i2w_bio
    }
    return vocab_dict

def get_feature_data(args, vocab_dict):
    """获得训练集和验证集"""
    logger.info("loading data...")

    data_processor_train = DataProcessor(
        os.path.join(args.data_dir, "train", "input.seq.char"),
        os.path.join(args.data_dir, "train", "output.seq.bio"),
        vocab_dict['w2i_char'],
        vocab_dict['w2i_bio'],
        shuffling=True
    )

    data_processor_valid = DataProcessor(
        os.path.join(args.data_dir, "dev", "input.seq.char"),
        os.path.join(args.data_dir, "dev", "output.seq.bio"),
        vocab_dict['w2i_char'],
        vocab_dict['w2i_bio'],
        shuffling=False
    )
    return data_processor_train, data_processor_valid

def get_predict_feature_data(args, vocab_dict):
    """获取测试集数据"""
    logger.info("loading predict data...")

    data_processor_test = DataProcessor_Test(
        os.path.join(args.test_input_file),
        vocab_dict['w2i_char'],
        vocab_dict['w2i_bio'],
        shuffling=False
    )
    return data_processor_test

def build_model(args, vocab_dict):
    """初始化模型"""
    logger.info("building model...")

    model = MyModel(embedding_dim=args.word_embedding_dim,
                    hidden_dim=args.encoder_hidden_dim,
                    vocab_size_char=len(vocab_dict['w2i_char']),
                    vocab_size_bio=len(vocab_dict['w2i_bio']),
                    use_crf=True)

    logger.info("model params:")
    params_num_all = 0
    for variable in tf.trainable_variables():
        params_num = 1
        for dim in variable.shape:
            params_num *= dim
        params_num_all += params_num
        logger.info("\t {} {} {}".format(variable.name, variable.shape, params_num))
    logger.info("all params num: " + str(params_num_all))
    return model

def evaluate(sess, model, data_processor, vocab_dict, max_batches=None, batch_size=1024):
    """验证/测试"""
    chars_seq = [] # 文本序列
    preds_seq = [] # 预测标签
    golds_seq = [] # 真实标签
    batches_sample = 0

    while True:
        (inputs_seq_batch,
         inputs_seq_len_batch,
         outputs_seq_batch,) = data_processor.get_batch(batch_size)

        feed_dict = {
            model.inputs_seq: inputs_seq_batch,
            model.inputs_seq_len: inputs_seq_len_batch,
            model.outputs_seq: outputs_seq_batch
        }

        preds_seq_batch = sess.run(model.outputs, feed_dict)

        for pred_seq, gold_seq, input_seq, l in zip(preds_seq_batch,
                                                    outputs_seq_batch,
                                                    inputs_seq_batch,
                                                    inputs_seq_len_batch):
            pred_seq = [vocab_dict['i2w_bio'][i] for i in pred_seq[:l]]
            gold_seq = [vocab_dict['i2w_bio'][i] for i in gold_seq[:l]]
            char_seq = [vocab_dict['i2w_char'][i] for i in input_seq[:l]]

            chars_seq.append(char_seq)
            preds_seq.append(pred_seq)
            golds_seq.append(gold_seq)

        if data_processor.end_flag:
            data_processor.refresh()
            break

        batches_sample += 1
        if (max_batches is not None) and (batches_sample >= max_batches):
            break
    return (chars_seq, preds_seq, golds_seq)


def predict_evaluate(sess, model, data_processor, vocab_dict, max_batches=None, batch_size=1024):
    """输出预测结果"""
    chars_seq = []  # 文本序列
    preds_seq = []  # 预测标签
    eids = []
    sids = []
    batches_sample = 0

    while True:
        (inputs_seq_batch,
         inputs_seq_len_batch,
         eids_batch,
         sids_batch) = data_processor.get_batch(batch_size)

        feed_dict = {
            model.inputs_seq: inputs_seq_batch,
            model.inputs_seq_len: inputs_seq_len_batch,
        }

        preds_seq_batch = sess.run(model.outputs, feed_dict)

        for pred_seq, input_seq, l, eid, sid in zip(preds_seq_batch,
                                                    inputs_seq_batch,
                                                    inputs_seq_len_batch,
                                                    eids_batch,
                                                    sids_batch):
            pred_seq = [vocab_dict['i2w_bio'][i] for i in pred_seq[:l]]
            char_seq = [vocab_dict['i2w_char'][i] for i in input_seq[:l]]

            chars_seq.append(char_seq)
            preds_seq.append(pred_seq)
            eids.append(eid)
            sids.append(sid)
        if data_processor.end_flag:
            data_processor.refresh()
            break

        batches_sample += 1
        if (max_batches is not None) and (batches_sample >= max_batches):
            break
    assert len(chars_seq) == len(preds_seq) == len(eids)
    return (chars_seq, preds_seq, eids, sids)


def train(tf_config, args, model, data_processor_train, data_processor_valid, vocab_dict):
    """训练"""
    logger.info("start training...")

    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=50)

        epoches = 0
        losses = []
        batches = 0
        best_f1 = 0
        batch_size = args.batch_size

        while epoches < args.num_epoch:
            # 获取batch
            (inputs_seq_batch,
             inputs_seq_len_batch,
             outputs_seq_batch) = data_processor_train.get_batch(batch_size)

            feed_dict = {
                model.inputs_seq: inputs_seq_batch,
                model.inputs_seq_len: inputs_seq_len_batch,
                model.outputs_seq: outputs_seq_batch
            }

            if batches == 0:
                logger.info("###### shape of a batch #######")
                logger.info("input_seq: " + str(inputs_seq_batch.shape))
                logger.info("input_seq_len: " + str(inputs_seq_len_batch.shape))
                logger.info("output_seq: " + str(outputs_seq_batch.shape))
                logger.info("###### preview a sample #######")
                logger.info("input_seq:" + " ".join([vocab_dict['i2w_char'][i] for i in inputs_seq_batch[0]]))
                logger.info("input_seq_len :" + str(inputs_seq_len_batch[0]))
                logger.info("output_seq: " + " ".join([vocab_dict['i2w_bio'][i] for i in outputs_seq_batch[0]]))
                logger.info("###############################")

            loss, _ = sess.run([model.loss, model.train_op], feed_dict)
            losses.append(loss)
            batches += 1

            if data_processor_train.end_flag:
                data_processor_train.refresh()
                epoches += 1

            # 一定次数后进行验证
            if batches % args.evaluate_steps == 0:
                logger.info("")
                logger.info("Epoches: {}".format(epoches))
                logger.info("Batches: {}".format(batches))
                logger.info("Loss: {}".format(sum(losses) / len(losses)))
                losses = []

                model_save_path = os.path.join(args.save_dir, "model.ckpt")
                saver.save(sess, model_save_path)
                logger.info("Path of model: {}".format(model_save_path))

                (_, preds_seq, golds_seq) = evaluate(sess, model, data_processor_valid, vocab_dict, max_batches=100, batch_size=1024)
                results = compute_metrics(preds_seq, golds_seq)
                p, r, f1 = results['precision'], results['recall'], results['f1']
                logger.info("Valid Samples: {}".format(len(preds_seq)))
                logger.info(
                    "Valid P/R/F1: {} / {} / {}".format(round(p * 100, 2), round(r * 100, 2), round(f1 * 100, 2)))

                if f1 > best_f1:
                    best_f1 = f1
                    logger.info("############# best performance now here ###############")
                    best_model_save_path = os.path.join(args.save_dir, "best_model.ckpt")
                    saver.save(sess, best_model_save_path)
                    logger.info("Path of best model: {}".format(best_model_save_path))
                    # logger.info("=========Testing===========")
                    # p, r, f1 = valid(data_processor_test, max_batches=100)


def predict(args, model, data_processor_test, vocab_dict):
    """预测并输出结果"""
    # meta_path = os.path.join(args.save_dir, 'best_model.ckpt.meta')
    ckpt_path = os.path.join(args.save_dir, 'best_model.ckpt')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        # saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, ckpt_path)

        (chars_seq, preds_seq, eids, sids) = predict_evaluate(sess, model, data_processor_test, vocab_dict, max_batches=100, batch_size=1024)
        # results = compute_metrics(preds_seq, golds_seq)
        # p, r, f1 = results['precision'], results['recall'], results['f1']
        # logger.info("Test Samples: {}".format(len(preds_seq)))
        # logger.info(
        #     "Test P/R/F1: {} / {} / {}".format(round(p * 100, 2), round(r * 100, 2), round(f1 * 100, 2)))

        # 保存结果
        outputs = {}
        for i in range(len(preds_seq)):
            pred_seq = preds_seq[i]
            eid = eids[i]
            sid = sids[i]
            if eid not in outputs:
                outputs[eid] = {}
                outputs[eid][sid] = ' '.join(pred_seq[3:])  # 只保留句子的BIO标签，删去了speaker的BIO标签
                # outputs[eid]['dialogue'].append({'sentence_id': sid, 'BIO_label': ' '.join(pred_seq)})
            else:
                outputs[eid][sid] = ' '.join(pred_seq[3:])

        pred_path = os.path.join(args.test_output_file)

        with open(pred_path, 'w', encoding='utf-8') as json_file:
            json.dump(outputs, json_file, ensure_ascii=False, indent=4)
        print('=========end prediction===========')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-dd', type=str, default='data/ner_data', help='Train/dev data path')
    parser.add_argument('--save_dir', '-sd', type=str, default='save_model', help='Path to save, load model')
    parser.add_argument('--test_input_file', '-tif', type=str, default='../../dataset/test.json', help='Input file for prediction')
    parser.add_argument('--test_output_file', '-tof', type=str, default='submission_track1_task1.json', help='Output file for prediction')
    parser.add_argument('--do_train', '-train', action='store_true', default=False, help='Whether to run training')
    parser.add_argument('--do_predict', '-predict', action='store_true', default=False, help='Whether to run predicting')
    parser.add_argument('--use_crf', '-crf', action='store_true', default=True, help='Whether to use CRF')
    parser.add_argument('--word_embedding_dim', '-wed', type=int, default=300, help='Word embedding dim')
    parser.add_argument('--encoder_hidden_dim', '-ehd', type=int, default=300, help='LSTM encoder hidden dim')
    parser.add_argument('--num_epoch', '-ne', type=int, default=10, help='Total number of training epochs to perform')
    parser.add_argument('--batch_size', '-bs', type=int, default=32, help='Batch size for trainging')
    parser.add_argument('--random_seed', '-rs', type=int, default=6, help='Random seed')
    parser.add_argument('--evaluate_steps', '-ls', type=int, default=200, help='Evaluate every X updates steps')

    args = parser.parse_args()

    random.seed(args.random_seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    init_logging(args)
    vocab_dict = get_vocab(args)
    model = build_model(args, vocab_dict)

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True

    if args.do_train:
        data_processor_train, data_processor_valid = get_feature_data(args, vocab_dict)
        train(tf_config, args, model, data_processor_train, data_processor_valid, vocab_dict)

    if args.do_predict:
        data_processor_test = get_predict_feature_data(args, vocab_dict)
        predict(args, model, data_processor_test, vocab_dict)
