# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.AttentionModelCore import AttentionModelCore


class LSTMSoftMultiAttentionFeatArrayNoInputCore(nn.Module):
    # att_feat_size and att_num are list
    def __init__(self, rnn_size, att_feat_size, att_num, att_hid_size, drop_prob_lm, maxout=0):
        super(LSTMSoftMultiAttentionFeatArrayNoInputCore, self).__init__()

        self.rnn_size = rnn_size
        self.att_feat_size = att_feat_size
        self.att_num = att_num
        assert(len(att_feat_size) == len(att_num))
        self.num_feat_array = len(att_feat_size)
        self.drop_prob_lm = drop_prob_lm

        self.att_hid_size = att_hid_size
        self.maxout = maxout

        # Build a LSTM
        if self.maxout:
            self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
            self.z_2_h = nn.ModuleList([nn.Linear(self.att_feat_size[i], 5 * self.rnn_size) for i in range(self.num_feat_array)])
        else:
            self.h2h = nn.Linear(self.rnn_size, 4 * self.rnn_size)
            self.z_2_h = nn.ModuleList([nn.Linear(self.att_feat_size[i], 4 * self.rnn_size) for i in range(self.num_feat_array)])

        self.att_model = nn.ModuleList([AttentionModelCore(rnn_size, att_feat_size[i], att_num[i], att_hid_size) for i in range(self.num_feat_array)])
        self.dropout = nn.Dropout(self.drop_prob_lm)

        # init
        initrange = 0.1
        self.h2h.weight.data.uniform_(-initrange, initrange)
        self.h2h.bias.data.uniform_(-initrange, initrange)

    # att_seq is a list
    def forward(self, att_seq, state):  # state = (pre_h, pre_c)
        assert(len(att_seq) == self.num_feat_array, 'att_seq: ' + str(len(att_seq)) + ', num_feat_array: ' + str(self.num_feat_array))
        pre_h = state[0][-1]
        pre_c = state[1][-1]

        z = [[] for i in range(self.num_feat_array)]
        for i in range(self.num_feat_array):
            z[i] = self.att_model[i](pre_h, att_seq[i])

        all_input_sums = self.h2h(pre_h)
        for i in range(self.num_feat_array):
            all_input_sums += self.z_2_h[i](z[i])

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
