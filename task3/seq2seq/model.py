from __future__ import unicode_literals, print_function, division

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from numpy import random


random.seed(111)
torch.manual_seed(111)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(111)

def init_lstm_wt(lstm,args):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-args.rand_unif_init_mag, args.rand_unif_init_mag)
            elif name.startswith('bias_'):
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear,args):
    linear.weight.data.normal_(std=args.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=args.trunc_norm_init_std)

def init_wt_normal(wt,args):
    wt.data.normal_(std=args.trunc_norm_init_std)

def init_wt_unif(wt,args):
    wt.data.uniform_(-args.rand_unif_init_mag, args.rand_unif_init_mag)

class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(self.args.vocab_size, self.args.emb_dim)
        init_wt_normal(self.embedding.weight,self.args)
        
        self.lstm = nn.LSTM(self.args.emb_dim, self.args.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm,self.args)

        self.W_h = nn.Linear(self.args.hidden_dim * 2, self.args.hidden_dim * 2, bias=False)

    #seq_lens should be in descending order
    def forward(self, input, seq_lens):
        embedded = self.embedding(input)
       
        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs.contiguous()
        
        encoder_feature = encoder_outputs.view(-1, 2*self.args.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        return encoder_outputs, encoder_feature, hidden

class ReduceState(nn.Module):
    def __init__(self,args):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        init_linear_wt(self.reduce_h,args)
        self.reduce_c = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        init_linear_wt(self.reduce_c,args)

        self.args = args

    def forward(self, hidden):
        h, c = hidden # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, self.args.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, self.args.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_dim

class Attention(nn.Module):
    def __init__(self,args):
        super(Attention, self).__init__()

        self.args = args
        if self.args.is_coverage:
            self.W_c = nn.Linear(1, self.args.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(self.args.hidden_dim * 2, self.args.hidden_dim * 2)
        self.v = nn.Linear(self.args.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        if self.args.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / (normalization_factor.view(-1,1) + torch.ones_like(normalization_factor.view(-1, 1)) * sys.float_info.epsilon)

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, self.args.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if self.args.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.args =args

        self.attention_network = Attention(args)
        # decoder
        self.embedding = nn.Embedding(self.args.vocab_size, self.args.emb_dim)
        init_wt_normal(self.embedding.weight,args)

        self.x_context = nn.Linear(self.args.hidden_dim * 2 + self.args.emb_dim, self.args.emb_dim)

        self.lstm = nn.LSTM(self.args.emb_dim, self.args.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm,args)

        if self.args.pointer_gen:
            self.p_gen_linear = nn.Linear(self.args.hidden_dim * 4 + self.args.emb_dim, 1)

        self.out1 = nn.Linear(self.args.hidden_dim * 3, self.args.hidden_dim)
        self.out2 = nn.Linear(self.args.hidden_dim, self.args.vocab_size)
        init_linear_wt(self.out2,args)

        self.args = args

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):

        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, self.args.hidden_dim),
                                 c_decoder.view(-1, self.args.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                              enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, self.args.hidden_dim),
                             c_decoder.view(-1, self.args.hidden_dim)), 1)  # B x 2*hidden_dim
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                          enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if self.args.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, self.args.hidden_dim), c_t), 1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim
        # 每个字的概率分布
        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if self.args.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage

class Model(object):
    def __init__(self, args, model_file_path=None, is_eval=False):
        encoder = Encoder(args)
        decoder = Decoder(args)
        reduce_state = ReduceState(args)

        self.args = args 
        #编码器和解码器的嵌入部分，权重相同
        decoder.embedding.weight = encoder.embedding.weight

        #decode.py中is_eval=True
        if is_eval:   
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        use_cuda = self.args.use_gpu and torch.cuda.is_available()
        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()

        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])
