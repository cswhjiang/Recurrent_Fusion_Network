# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

import opts


class LSTMSoftAttentionNoInputCore(nn.Module):
    def __init__(self, rnn_size, att_feat_size, att_num, att_hid_size, drop_prob_reason, maxout=0):
        super(LSTMSoftAttentionNoInputCore, self).__init__()

        self.rnn_size = rnn_size
        self.drop_prob_reason = drop_prob_reason  #
        self.att_feat_size = att_feat_size
        self.att_num = att_num
        self.att_hid_size = att_hid_size
        self.maxout = maxout

        # Build a LSTM
        if self.maxout:
            self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
            self.z2h = nn.Linear(self.att_feat_size, 5 * self.rnn_size)
        else:
            self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
            self.z2h = nn.Linear(self.att_feat_size, 4 * self.rnn_size)

        # for soft attention
        self.att_2_att_h = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.h_2_att_h = nn.Linear(self.rnn_size, self.att_hid_size)
        self.att_h_2_out = nn.Linear(self.att_hid_size, 1)

        self.dropout = nn.Dropout(self.drop_prob_reason)

        # init
        initrange = 0.1
        self.h2h.weight.data.uniform_(-initrange, initrange)
        self.h2h.bias.data.fill_(-1)
        self.z2h.weight.data.uniform_(-initrange, initrange)
        self.z2h.bias.data.fill_(-1)

        self.att_2_att_h.weight.data.uniform_(-initrange, initrange)
        self.att_2_att_h.bias.data.fill_(0)

        self.h_2_att_h.weight.data.uniform_(-initrange, initrange)
        self.h_2_att_h.bias.data.fill_(0)

        self.att_h_2_out.weight.data.uniform_(-initrange, initrange)
        self.att_h_2_out.bias.data.fill_(0)

    def forward(self, att_seq, mil_feats, matching_feats, state):  # state = (pre_h, pre_c)
        pre_h = state[0][-1]
        pre_c = state[1][-1]
        # print(self.h2h.weight.norm() + self.z2h.weight.norm())

        att = att_seq.view(-1, self.att_feat_size)
        att_linear = self.att_2_att_h(att)  # att_num * att_hid_size
        att_linear = att_linear.view(-1, self.att_num, self.att_hid_size)  # batch * att_num * att_hid_size

        h_linear = self.h_2_att_h(pre_h)  # batch*att_hid_size
        h_linear_expand = h_linear.unsqueeze(0).expand(self.att_num, h_linear.size()[0], h_linear.size()[1]).transpose(
            0, 1)

        att_h = F.tanh(h_linear_expand + att_linear)

        att_h_view = att_h.contiguous().view(-1, self.att_hid_size)
        att_out = self.att_h_2_out(att_h_view)
        att_out_view = att_out.view(-1, self.att_num)
        conv_weight = nn.Softmax()(att_out_view)  # batch * att_size
        conv_weight_unsqueeze = conv_weight.unsqueeze(2)  # batch * att_size * 1
        att_seq_t = att_seq.transpose(1, 2)  # batch * feat_size * att_size
        z = torch.bmm(att_seq_t, conv_weight_unsqueeze).squeeze()

        all_input_sums = self.h2h(pre_h) + self.z2h(z)

        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk_sig = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk_sig.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk_sig.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk_sig.narrow(1, self.rnn_size * 2, self.rnn_size)

        if self.maxout:
            in_transform = torch.max(all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size),
                                     all_input_sums.narrow(1, 4 * self.rnn_size, self.rnn_size))
        else:
            in_transform = F.tanh(all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size))

        next_c = forget_gate * pre_c + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        next_h = self.dropout(next_h)

        output = next_h
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state


if __name__ == '__main__':
    opt = opts.parse_opt()
    model = LSTMSoftAttentionNoInputCore(opt)
    # xt = Variable(torch.randn(opt.batch_size, opt.input_encoding_size))
    att_seq = Variable(torch.randn(opt.batch_size, opt.att_num, opt.att_feat_size))
    pre_c = Variable(torch.randn(opt.batch_size, opt.rnn_size))
    pre_h = Variable(torch.randn(opt.batch_size, opt.rnn_size))
    state = (pre_h, pre_c)
    out, newstate = model(att_seq, state)
    print(out.sum())
