# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModelCore(nn.Module):
    def __init__(self, rnn_size, att_feat_size, att_num, att_hid_size):
        super(AttentionModelCore, self).__init__()
        self.rnn_size = rnn_size
        self.att_hid_size = att_hid_size
        self.att_feat_size = att_feat_size
        self.att_num = att_num

        self.att_2_att_h = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.h_2_att_h = nn.Linear(self.rnn_size, self.att_hid_size)
        self.att_h_2_out = nn.Linear(self.att_hid_size, 1)

        # init
        initrange = 0.1
        self.att_2_att_h.weight.data.uniform_(-initrange, initrange)
        self.att_2_att_h.bias.data.uniform_(-initrange, initrange)

        self.h_2_att_h.weight.data.uniform_(-initrange, initrange)
        self.h_2_att_h.bias.data.uniform_(-initrange, initrange)

        self.att_h_2_out.weight.data.uniform_(-initrange, initrange)
        self.att_h_2_out.bias.data.uniform_(-initrange, initrange)

    def forward(self, pre_h, att_seq):  # state = (pre_h, pre_c)
        att = att_seq.view(-1, self.att_feat_size)
        att_linear = self.att_2_att_h(att)  # (batch* att_num) * att_hid_size
        att_linear = att_linear.view(-1, self.att_num, self.att_hid_size)  # batch * att_num * att_hid_size

        h_linear = self.h_2_att_h(pre_h)  # batch*att_hid_size
        h_linear_expand = h_linear.unsqueeze(0).expand(self.att_num, h_linear.size(0), h_linear.size(1)).transpose(0, 1)

        att_h = F.tanh(h_linear_expand + att_linear)

        att_h_view = att_h.contiguous().view(-1, self.att_hid_size)
        att_out = self.att_h_2_out(att_h_view)
        att_out_view = att_out.view(-1, self.att_num)
        conv_weight = nn.Softmax()(att_out_view)  # batch * att_size
        conv_weight_unsqueeze = conv_weight.unsqueeze(2)  # batch * att_size * 1
        att_seq_t = att_seq.transpose(1, 2)  # batch * feat_size * att_size
        z = torch.bmm(att_seq_t, conv_weight_unsqueeze).squeeze()
        return z
