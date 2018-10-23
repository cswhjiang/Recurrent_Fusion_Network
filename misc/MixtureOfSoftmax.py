# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


# ref: Breaking the Softmax Bottleneck: A High-Rank RNN Language Model (https://arxiv.org/abs/1711.03953)
class MixtureOfSoftmax(nn.Module):
    def __init__(self, rnn_size, emb_size, n_experts, dict_size):
        super(MixtureOfSoftmax, self).__init__()
        self.rnn_size = rnn_size
        self.emb_size = emb_size
        self.n_experts = n_experts
        self.dict_size = dict_size

        self.prior = nn.Linear(rnn_size, n_experts, bias=False)
        # self.latent = nn.Sequential(nn.Linear(rnn_size, n_experts * emb_size), nn.Tanh())
        self.latent = nn.ModuleList([nn.Sequential(nn.Linear(rnn_size,  emb_size), nn.Tanh()) for _ in range(self.n_experts)])
        self.decoder = nn.Linear(emb_size, dict_size)

    def forward(self, output):
        prior_logit = self.prior(output)
        prior = F.softmax(prior_logit)

        prob_list = []
        for i in range(self.n_experts):
            prob = F.softmax(self.decoder(self.latent[i](output))) * prior[:, i].unsqueeze(1)
            prob_list.append(prob)

        prob_all = sum(prob_list)

        return prob_all


if __name__ == '__main__':
    rnn_size = 100
    emb_size = 50
    n_experts = 10
    dict_size = 200
    batch_size = 16
    output = Variable(torch.randn(batch_size, rnn_size))
    mos = MixtureOfSoftmax(rnn_size, emb_size, n_experts, dict_size)
    log_prob = mos(output)
    print(log_prob.size())
    # print(sum(torch.exp(log_prob)))
    print(torch.sum(log_prob, 1))
