from __future__ import unicode_literals, print_function, division

import sys
import argparse
import os
import time
import numpy as np
import torch
from torch.autograd import Variable
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util import data  
from model import Model
from data_util.utils import evaluate_from_decoded,write_for_rouge,rouge_log

def get_input_from_batch(args,batch, use_cuda):
    batch_size = len(batch.enc_lens)

    enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
    enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()
    enc_lens = batch.enc_lens
    extra_zeros = None
    enc_batch_extend_vocab = None

    if args.pointer_gen:
        enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
        # max_art_oovs is the max over all the dialogue oov list in the batch
        if batch.max_art_oovs > 0:
            extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))

    c_t_1 = Variable(torch.zeros((batch_size, 2 * args.hidden_dim)))

    coverage = None
    if args.is_coverage:
        coverage = Variable(torch.zeros(enc_batch.size()))

    if use_cuda:
        enc_batch = enc_batch.cuda()
        enc_padding_mask = enc_padding_mask.cuda()

        if enc_batch_extend_vocab is not None:
            enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()
        c_t_1 = c_t_1.cuda()

        if coverage is not None:
            coverage = coverage.cuda()

    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage


class Beam(object):
  def __init__(self, tokens, log_probs, state, context, coverage):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage

  def extend(self, token, log_prob, state, context, coverage):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      context = context,
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, args):
        
        self.args = args
        model_name = os.path.basename(self.args.model_filename)
        self._decode_dir = os.path.join(self.args.log_root, 'decode_%s' % (model_name))
        self._rouge_dec_dir = os.path.join(self._decode_dir, '生成摘要_'+self.args.mode)
        self._rouge_ref_dir = os.path.join(self._decode_dir, '参考摘要_'+self.args.mode)
        for p in [self._decode_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)
        if args.compute_rouge:
            if not os.path.exists(self._rouge_ref_dir):
                os.mkdir(self._rouge_ref_dir)

        self.vocab = Vocab(vocab_path, self.args.vocab_size)
        self.batcher = Batcher(self.args, decode_data_path, self.vocab, mode='decode',
                               batch_size=self.args.beam_size, single_pass=True)
        time.sleep(15)

        self.model = Model(self.args, self.args.model_filename, is_eval=True)

        self.use_cuda = self.args.use_gpu and torch.cuda.is_available()

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


    def decode(self):
        start = time.time()
        counter = 0
        batch = self.batcher.next_batch()
        decode_file_names = output_filenames 
        with open(decode_file_names) as f:
            f_names = f.read().splitlines()
        while batch is not None:
            
            best_summary = self.beam_search(batch)
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self.vocab,
                                                 (batch.art_oovs[0] if self.args.pointer_gen else None))

            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_abstract_sents = batch.original_abstracts_sents[0]

            write_for_rouge(original_abstract_sents, decoded_words, f_names[counter],
                            self._rouge_ref_dir, self._rouge_dec_dir,self.args.compute_rouge)
            counter += 1
            if counter % 100 == 0:
                print('%d example in %d sec'%(counter, time.time() - start))
                start = time.time()

            batch = self.batcher.next_batch()

        print("Decoder has finished reading dataset for single_pass.")
        
        if self.args.compute_rouge==True:
            print("Now starting ROUGE eval...")
            mean_scores = evaluate_from_decoded(self._rouge_ref_dir, self._rouge_dec_dir, self.vocab._word_to_id)
            rouge_log(mean_scores, self._decode_dir)
            
            print('计算出验证集的R-1，R-2，R-l结果为：',mean_scores)


    def beam_search(self, batch):
        '''通过beam search获取最佳预测结果
        # 阶段阶段，每个batch只含有一组对话'''
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
            get_input_from_batch(self.args,batch, self.use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = c_t_0[0],
                      coverage=(coverage_t_0[0] if self.args.is_coverage else None))
                 for _ in range(self.args.beam_size)]
        results = []
        steps = 0
        while steps < self.args.max_dec_steps and len(results) < self.args.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if self.use_cuda:
                y_t_1 = y_t_1.cuda()
            all_state_h =[]
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if self.args.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, self.args.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if self.args.is_coverage else None)

                for j in range(self.args.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= self.args.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == self.args.beam_size or len(results) == self.args.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]



if __name__ == '__main__':
    current_dir = os.getcwd()
    parser = argparse.ArgumentParser(description="Decode")

    parser.add_argument("--hidden_dim", default=256, type=int ,
                        help="隐藏层向量维度")
    parser.add_argument("--emb_dim", default=128, type=int ,
                        help="嵌入向量维度")
    parser.add_argument("--batch_size", default=8, type=int ,
                        help="训练时每个batch的大小")
    parser.add_argument("--max_enc_steps", default=1024, type=int ,
                        help="输入对话文本的最大长度，超过该程度进行截断")
    parser.add_argument("--max_dec_steps", default=200, type=int ,
                        help="输出诊疗报告的最大长度")
    parser.add_argument("--min_dec_steps", default=50, type=int ,
                        help="输出诊疗报告的最小长度")
    parser.add_argument("--beam_size", default=4, type=int ,
                        help="beam search的大小")
    parser.add_argument("--vocab_size", default=3000, type=int ,
                        help="词典大小")
    
    parser.add_argument("--lr", default=0.15, type=float ,
                        help="学习率")
    parser.add_argument("--adagrad_init_acc", default=0.1, type=float ,
                        help="Adagrad的初始累加器值")
    parser.add_argument("--rand_unif_init_mag", default=0.02, type=float ,
                        help="lstm单元随机均匀初始化的幅度")
    parser.add_argument("--trunc_norm_init_std", default=1e-4, type=float ,
                        help="张量初始化标准差")
    parser.add_argument("--max_grad_norm", default=2.0, type=float ,
                        help="梯度的最大范数")       

    parser.add_argument("--eps", default=1e-12, type=float ,
                        help="epsilon")      

    parser.add_argument("--pointer_gen", action="store_true",default=False,
                        help="是否使用指针生成器，默认为否")   
    parser.add_argument("--is_coverage", action="store_true",default=False,
                        help="是否使用汇聚机制，默认为否")   
    parser.add_argument("--cov_loss_wt", default=1.0, type=float ,
                        help="汇聚机制对应的损失权重")  
    parser.add_argument("--lr_coverage", default=0.15, type=float ,
                        help="汇聚机制下，进行训练的学习率")  
    parser.add_argument("--max_iterations", default=60000, type=int ,
                        help="训练过程中，最大的迭代次数")                                             
    parser.add_argument("--use_gpu", action="store_true",default=False,
                        help="是否使用gpu，默认为否")  
    parser.add_argument("--log_root", default=os.path.join(current_dir, "log/"), type=str,
                        help="log的位置")  
    parser.add_argument("--exp_name", default="exp_1", type=str,
                        help="实验名称")  
    parser.add_argument("--model_filename",  required=True, type=str,
                        help="模型名称")  

    parser.add_argument("--decode_filename", default="medi_finished_dir/test.bin", type=str,
                        help="需要生成摘要的文件") 
    parser.add_argument("--output_filenames", default="medi_finished_dir/file_names_test", type=str,
                        help="输出的文件名存在该文件中")

    parser.add_argument("--compute_rouge", action="store_true",default=False,
                        help="是否计算ROUGE值")

    parser.add_argument("--mode", default="test", type=str,
                        help="是否计算ROUGE值")
                    
    args = parser.parse_args()
    
    decode_data_path = os.path.join(current_dir, args.decode_filename)

    vocab_path = os.path.join(current_dir, "medi_finished_dir/vocab")
    output_filenames = os.path.join(current_dir,args.output_filenames)
    
    beam_Search_processor = BeamSearch(args)
    beam_Search_processor.decode()



