from __future__ import unicode_literals, print_function, division

import os
import time
import argparse

import numpy as np
import tensorflow as tf
import torch
from model import Model
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adagrad


from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss


current_dir = os.getcwd()

train_data_path = os.path.join(current_dir, "medi_finished_dir/chunked/train_*")
eval_data_path = os.path.join(current_dir, "medi_finished_dir/dev.bin")
decode_data_path = os.path.join(current_dir, "medi_finished_dir/test.bin")
vocab_path = os.path.join(current_dir, "medi_finished_dir/vocab")




class Train(object):
    def __init__(self,args):
        self.vocab = Vocab(vocab_path, args.vocab_size)
        self.batcher = Batcher(args, train_data_path, self.vocab, mode='train',
                               batch_size=args.batch_size, single_pass=False)
        time.sleep(15)

        self.args = args
        train_dir = os.path.join(args.log_root, 'train_'+args.exp_name)  #%d' % (int(time.time())))
        if not os.path.exists(args.log_root):
            os.mkdir(args.log_root)
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.summary.FileWriter(train_dir)

    def save_model(self, running_avg_loss, iter):
        '''保存模型'''
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        self.model = Model(self.args,model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        initial_lr = args.lr_coverage if args.is_coverage else args.lr
        self.optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=args.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not args.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch):
        '''训练一个batch'''
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            self.get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            self.get_output_from_batch(batch, use_cuda)

        self.optimizer.zero_grad()

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, args.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab,
                                                                           coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + args.eps)
            if args.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + args.cov_loss_wt * step_coverage_loss
                coverage = next_coverage
                
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), args.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), args.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), args.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def trainIters(self, n_iters, model_file_path=None):
        '''迭代训练'''
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % 100 == 0:
                self.summary_writer.flush()
            print_interval = 100
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                           time.time() - start, loss))
                start = time.time()
            if iter % 500 == 0:
                tmp_min_loss = loss
                print('模型已保存')
                self.save_model(running_avg_loss, iter)
    

    def get_input_from_batch(self,batch, use_cuda):
        '''获取一个batch的输入'''
        batch_size = len(batch.enc_lens)

        enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
        enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()
        enc_lens = batch.enc_lens
        extra_zeros = None
        enc_batch_extend_vocab = None

        if args.pointer_gen:
            enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
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

    def get_output_from_batch(self,batch, use_cuda):
        '''获取一个batch的输出'''
        dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
        dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()
        dec_lens = batch.dec_lens
        max_dec_len = np.max(dec_lens)
        dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()

        target_batch = Variable(torch.from_numpy(batch.target_batch)).long()

        if use_cuda:
            dec_batch = dec_batch.cuda()
            dec_padding_mask = dec_padding_mask.cuda()
            dec_lens_var = dec_lens_var.cuda()
            target_batch = target_batch.cuda()
        return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("-m",  dest="model_file_path", required=False,  default=None, 
                        help="Model file for retraining (default: None).")
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
    parser.add_argument("--max_iterations", default=10000, type=int ,
                        help="训练过程中，最大的迭代次数")                                             
    parser.add_argument("--use_gpu", action="store_true",default=False,
                        help="是否使用gpu，默认为否")  
    parser.add_argument("--log_root", default=os.path.join(current_dir, "log/"), type=str,
                        help="log的位置")  
    parser.add_argument("--exp_name", default="exp_1", type=str,
                        help="实验名称")  
                        

    args = parser.parse_args()
    
    use_cuda = args.use_gpu and torch.cuda.is_available()
    train_processor = Train(args)
    print('--------开始训练-------')
    train_processor.trainIters(args.max_iterations, args.model_file_path)





