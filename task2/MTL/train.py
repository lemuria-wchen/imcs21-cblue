import logging
import tensorflow as tf
import os
import argparse
import random
import json

from collections import defaultdict
from model import MyModel
from utils import DataProcessor_MTL_LSTM as DataProcessor
from utils import DataProcessor_MTL_LSTM_Test as DataProcessor_Test
from utils import load_vocabulary
from utils import extract_kvpairs_in_bioes_type
from utils import cal_f1_score


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
    w2i_char, i2w_char = load_vocabulary(os.path.join(args.data_dir, "vocab_char.txt"))  # 单词表
    w2i_bio, i2w_bio = load_vocabulary(os.path.join(args.data_dir, "vocab_bio.txt"))  # BIO表
    w2i_attr, i2w_attr = load_vocabulary(os.path.join(args.data_dir, "vocab_attr.txt"))  # 实体归一化 [咳嗽 咳嗽 null null null null]
    w2i_type, i2w_type = load_vocabulary(os.path.join(args.data_dir, "vocab_type.txt")) # 实体属性 [1 1 null null null null null]
    vocab_dict = {
        "w2i_char": w2i_char,
        "i2w_char": i2w_char,
        "w2i_bio": w2i_bio,
        "i2w_bio": i2w_bio,
        "w2i_attr": w2i_attr,
        "i2w_attr": i2w_attr,
        "w2i_type": w2i_type,
        "i2w_type": i2w_type
    }
    return vocab_dict

def get_feature_data(args, vocab_dict):
    """获得训练集和验证集"""
    # 转换成digit
    data_processor_train = DataProcessor(
        os.path.join(args.data_dir, "train", "input.seq.char"),
        os.path.join(args.data_dir, "train", "output.seq.bio"),
        os.path.join(args.data_dir, "train", "output.seq.attr"),
        os.path.join(args.data_dir, "train", "output.seq.type"),
        vocab_dict['w2i_char'],
        vocab_dict['w2i_bio'],
        vocab_dict['w2i_attr'],
        vocab_dict['w2i_type'],
        shuffling=True
    )

    # 转换成digit
    data_processor_valid = DataProcessor(
        os.path.join(args.data_dir, "dev", "input.seq.char"),
        os.path.join(args.data_dir, "dev", "output.seq.bio"),
        os.path.join(args.data_dir, "dev", "output.seq.attr"),
        os.path.join(args.data_dir, "dev", "output.seq.type"),
        vocab_dict['w2i_char'],
        vocab_dict['w2i_bio'],
        vocab_dict['w2i_attr'],
        vocab_dict['w2i_type'],
        shuffling=True
    )

    return data_processor_train, data_processor_valid

def get_predict_feature_data(args, vocab_dict):
    data_processor_test = DataProcessor_Test(
        os.path.join(args.test_input_file),
        vocab_dict['w2i_char'],
        vocab_dict['w2i_bio'],
        vocab_dict['w2i_attr'],
        vocab_dict['w2i_type'],
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
                    vocab_size_attr=len(vocab_dict['w2i_attr']),
                    vocab_size_type=len(vocab_dict['w2i_type']),
                    O_tag_index=vocab_dict['w2i_bio']["O"],
                    use_crf=args.use_crf) #改

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
    preds_kvpair = []
    golds_kvpair = []
    batches_sample = 0

    while True:
        (inputs_seq_batch,
         inputs_seq_len_batch,
         outputs_seq_bio_batch,
         outputs_seq_attr_batch,
         outputs_seq_type_batch,) = data_processor.get_batch(batch_size)

        feed_dict = {
            model.inputs_seq: inputs_seq_batch,
            model.inputs_seq_len: inputs_seq_len_batch,
            model.outputs_seq_bio: outputs_seq_bio_batch,
            model.outputs_seq_attr: outputs_seq_attr_batch,
            model.outputs_seq_type: outputs_seq_type_batch,
        }

        preds_seq_bio_batch, preds_seq_attr_batch, preds_seq_type_batch = sess.run(model.outputs, feed_dict)

        for pred_seq_bio, gold_seq_bio, pred_seq_attr, gold_seq_attr, \
            pred_seq_type, gold_seq_type, input_seq, l in zip(preds_seq_bio_batch,
                                                              outputs_seq_bio_batch,
                                                              preds_seq_attr_batch,
                                                              outputs_seq_attr_batch,
                                                              preds_seq_type_batch,
                                                              outputs_seq_type_batch,
                                                              inputs_seq_batch,
                                                              inputs_seq_len_batch):
            pred_seq_bio = [vocab_dict['i2w_bio'][i] for i in pred_seq_bio[:l]]
            gold_seq_bio = [vocab_dict['i2w_bio'][i] for i in gold_seq_bio[:l]]
            char_seq = [vocab_dict['i2w_char'][i] for i in input_seq[:l]]
            pred_seq_attr = [vocab_dict['i2w_attr'][i] for i in pred_seq_attr[:l]]
            gold_seq_attr = [vocab_dict['i2w_attr'][i] for i in gold_seq_attr[:l]]
            pred_seq_type = [vocab_dict['i2w_type'][i] for i in pred_seq_type[:l]]
            gold_seq_type = [vocab_dict['i2w_type'][i] for i in gold_seq_type[:l]]
            pred_kvpair = extract_kvpairs_in_bioes_type(pred_seq_bio, char_seq, pred_seq_attr,
                                                        pred_seq_type)  # (attr,type,word)
            gold_kvpair = extract_kvpairs_in_bioes_type(gold_seq_bio, char_seq, gold_seq_attr, gold_seq_type)  #

            preds_kvpair.append(pred_kvpair)
            golds_kvpair.append(gold_kvpair)  # {(attrs,types, words)}

        if data_processor.end_flag:
            data_processor.refresh()
            break

        batches_sample += 1
        if (max_batches is not None) and (batches_sample >= max_batches):
            break

    return (preds_kvpair, golds_kvpair)

def predict_evaluate(sess, model, data_processor, vocab_dict, max_batches=None, batch_size=1024):
    chars_seq = []
    preds_kvpair = []
    eids = []
    batches_sample = 0

    while True:
        (inputs_seq_batch,
         inputs_seq_len_batch,
         eids_batch) = data_processor.get_batch(batch_size)

        feed_dict = {
            model.inputs_seq: inputs_seq_batch,
            model.inputs_seq_len: inputs_seq_len_batch
        }

        preds_seq_bio_batch, preds_seq_attr_batch, preds_seq_type_batch = sess.run(model.outputs, feed_dict)

        for pred_seq_bio, pred_seq_attr, pred_seq_type, input_seq, l, eid in zip(preds_seq_bio_batch,
                                                              preds_seq_attr_batch,
                                                              preds_seq_type_batch,
                                                              inputs_seq_batch,
                                                              inputs_seq_len_batch,
                                                              eids_batch):

            pred_seq_bio = [vocab_dict['i2w_bio'][i] for i in pred_seq_bio[:l]]
            pred_seq_attr = [vocab_dict['i2w_attr'][i] for i in pred_seq_attr[:l]]
            pred_seq_type = [vocab_dict['i2w_type'][i] for i in pred_seq_type[:l]]
            char_seq = [vocab_dict['i2w_char'][i] for i in input_seq[:l]]

            pred_kvpair = extract_kvpairs_in_bioes_type(pred_seq_bio, char_seq, pred_seq_attr,
                                                        pred_seq_type)  # (attr,type,word)

            preds_kvpair.append(pred_kvpair) # {(attrs,types, words)}
            chars_seq.append(char_seq)
            eids.append(eid)

        if data_processor.end_flag:
            data_processor.refresh()
            break

        batches_sample += 1
        if (max_batches is not None) and (batches_sample >= max_batches):
            break
    assert len(chars_seq) == len(preds_kvpair) == len(eids)
    return (chars_seq, preds_kvpair, eids)


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
            (inputs_seq_batch,  #（B, T）
             inputs_seq_len_batch,  # 每句话的真实长度
             outputs_seq_bio_batch,  #（B, T）
             outputs_seq_attr_batch,  # (B, T)
             outputs_seq_type_batch) = data_processor_train.get_batch(batch_size)  #（B, T）

            feed_dict = {
                model.inputs_seq: inputs_seq_batch,
                model.inputs_seq_len: inputs_seq_len_batch,
                model.outputs_seq_bio: outputs_seq_bio_batch,
                model.outputs_seq_attr: outputs_seq_attr_batch,
                model.outputs_seq_type: outputs_seq_type_batch,
            }

            if batches == 0:
                logger.info("###### shape of a batch #######")
                logger.info("input_seq: " + str(inputs_seq_batch.shape))
                logger.info("input_seq_len: " + str(inputs_seq_len_batch.shape))
                logger.info("output_seq_bio: " + str(outputs_seq_bio_batch.shape))
                logger.info("output_seq_attr: " + str(outputs_seq_attr_batch.shape))
                logger.info("output_seq_type: " + str(outputs_seq_type_batch.shape))
                logger.info("###### preview a sample #######")
                logger.info("input_seq:" + " ".join([vocab_dict['i2w_char'][i] for i in inputs_seq_batch[0]]))
                logger.info("input_seq_len :" + str(inputs_seq_len_batch[0]))
                logger.info("output_seq_bio: " + " ".join([vocab_dict['i2w_bio'][i] for i in outputs_seq_bio_batch[0]]))
                logger.info("output_seq_attr: " + " ".join([vocab_dict['i2w_attr'][i] for i in outputs_seq_attr_batch[0]]))
                logger.info("output_seq_type: " + " ".join([vocab_dict['i2w_type'][i] for i in outputs_seq_type_batch[0]]))
                logger.info("###############################")

            loss, _ = sess.run([model.loss, model.train_op], feed_dict)
            losses.append(loss)
            batches += 1

            if data_processor_train.end_flag:
                data_processor_train.refresh()
                epoches += 1

            if batches % 100 == 0:
                logger.info("")
                logger.info("Epoches: {}".format(epoches))
                logger.info("Batches: {}".format(batches))
                logger.info("Loss: {}".format(sum(losses) / len(losses)))
                losses = []

                # ckpt_save_path = os.path.join(args.save_dir, "model.ckpt.batch{}".format(batches))
                model_save_path = os.path.join(args.save_dir, "model.ckpt")
                saver.save(sess, model_save_path)
                logger.info("Path of model: {}".format(model_save_path))

                (preds_kvpair, golds_kvpair) = evaluate(sess, model, data_processor_valid, vocab_dict, max_batches=100, batch_size=1024)
                p, r, f1 = cal_f1_score(preds_kvpair, golds_kvpair)
                logger.info("Valid Samples: {}".format(len(preds_kvpair)))
                logger.info("Valid P/R/F1: {} / {} / {}".format(round(p * 100, 2), round(r * 100, 2), round(f1 * 100, 2)))

                if f1 > best_f1:
                    best_f1 = f1
                    logger.info("############# best performance now here ###############")
                    best_model_save_path = os.path.join(args.save_dir, "best_model.ckpt")
                    saver.save(sess, best_model_save_path)
                    logger.info("Path of best model: {}".format(best_model_save_path))
                    # logger.info("=========Testing===========")
                    # p, r, f1 = valid(data_processor_test, max_batches=20)



def predict(args, model, data_processor_test, vocab_dict):
    """预测并输出结果"""
    # meta_path = os.path.join(args.save_dir, 'best_model.ckpt.meta')
    ckpt_path = os.path.join(args.save_dir, 'best_model.ckpt')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        # saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, ckpt_path)
        (chars_seq, preds_kvpair, eids) = predict_evaluate(sess, model, data_processor_test, vocab_dict, max_batches=100, batch_size=1024)
       
        # 将相同example id的数据以一定规则进行整合，得到样本级别的症状识别结果，用于评估。
        outputs = defaultdict(list)
        for i in range(len(eids)):
            if len(preds_kvpair[i]) != 0:
                outputs[eids[i]].extend(preds_kvpair[i])
        for eid, pairs in outputs.items():
            tmp_pred = defaultdict(list)
            if len(pairs) != 0:
                for pair in pairs:
                    tmp_pred[pair[0]].append(pair[1])
            for k, v in tmp_pred.items():
                new_v = max(v, key=v.count)
                tmp_pred[k] = new_v
            # 如果key 或 value为 null，则删除
            tmp_pred_new = {}
            for k, v in tmp_pred.items():
                if k != 'null' and v != 'null':
                    tmp_pred_new[k] = v
            outputs[eid] = tmp_pred_new
        # 将那些预测为空的样本id也存入进来，防止输出的样本缺失
        for eid in eids:
            if eid not in outputs:
                outputs[eid] = {}
        print("测试样本数量为：", len(outputs))
        pred_path = os.path.join(args.test_output_file)

        with open(pred_path, 'w', encoding='utf-8') as json_file:
            json.dump(outputs, json_file, ensure_ascii=False, indent=4)
        print('=========end prediction===========')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-dd', type=str, default='data/near_data', help='Train/dev data path')
    parser.add_argument('--save_dir', '-sd', type=str, default='save_model', help='Path to save, load model')
    parser.add_argument('--test_input_file', '-tif', type=str, default='../../dataset/test.json', help='Input file for prediction')
    parser.add_argument('--test_output_file', '-tof', type=str, default='submission_track1_task2.json', help='Output file for prediction')
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
