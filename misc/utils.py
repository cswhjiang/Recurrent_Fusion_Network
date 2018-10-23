# -*- coding: utf-8 -*-

import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def repackage(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage(v) for v in h)


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


# ---- rl reward
class ReviewNetRewardCriterion(nn.Module):
    def __init__(self, opt):
        super(ReviewNetRewardCriterion, self).__init__()
        self.use_label_smoothing = opt.use_label_smoothing
        self.label_smoothing_epsilon = opt.label_smoothing_epsilon

    def forward(self, input, seq, reward, logprobs_all, entropy_reg, top_pred, top_true, reason_weight, sample_logprobs_old, opt):
        batch_size = input.size(0)
        input_length = input.size(1)
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask_0 = (seq > 0).float()
        mask = Variable(to_contiguous(torch.cat([mask_0.new(mask_0.size(0), 1).fill_(1), mask_0[:, :-1]], 1)))
        mask = mask.view(-1)

        logprobs_all = logprobs_all[:, :input_length, :]
        temp = torch.sum(logprobs_all * torch.exp(logprobs_all), 2).squeeze()
        entropy_minus = temp * Variable(mask_0)
        if opt.use_ppo:
            probs = torch.exp(input)
            probs_old = torch.exp(sample_logprobs_old)
            ratio = probs / (1e-5 + probs_old)
            # clip loss
            surr1 = ratio * reward  # surrogate from conservative policy iteration
            surr2 = surr1.clamp(1 - opt.ppo_clip, 1 + opt.ppo_clip) * reward
            output = -torch.min(surr1, surr2) * mask
        else:
            output = - input * reward * mask
        output = torch.sum(output) / batch_size + entropy_reg * torch.sum(entropy_minus) / batch_size

        if not isinstance(top_pred, list):
            discriminative_loss = nn.MultiLabelMarginLoss()(top_pred, top_true)
            output = output + discriminative_loss * reason_weight
        else:
            discriminative_loss = []
            for i in range(len(top_pred)):
                discriminative_loss.append(nn.MultiLabelMarginLoss()(top_pred[i], top_true))

            output = output + sum(discriminative_loss) * reason_weight/len(top_pred)

        return output


class RewardCriterion(nn.Module):
    def __init__(self, opt):
        super(RewardCriterion, self).__init__()
        self.use_label_smoothing = opt.use_label_smoothing
        self.label_smoothing_epsilon = opt.label_smoothing_epsilon

    def forward(self, input, seq, reward, logprobs_all, entropy_reg, sample_logprobs_old, opt):
        batch_size = input.size(0)
        input_length = input.size(1)
        input = to_contiguous(input).view(-1)
        sample_logprobs_old = to_contiguous(sample_logprobs_old).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask_0 = (seq > 0).float()
        mask = Variable(to_contiguous(torch.cat([mask_0.new(mask_0.size(0), 1).fill_(1), mask_0[:, :-1]], 1)))
        mask = mask.view(-1)

        logprobs_all = logprobs_all[:, :input_length, :]
        temp = torch.sum(logprobs_all * torch.exp(logprobs_all), 2).squeeze()
        entropy_minus = temp * Variable(mask_0)
        if opt.use_ppo:
            probs = torch.exp(input)
            probs_old = torch.exp(sample_logprobs_old)
            ratio = probs / (1e-5 + probs_old)
            surr1 = ratio * reward  # surrogate from conservative policy iteration
            surr2 = ratio.clamp(1.0 - opt.ppo_clip, 1.0 + opt.ppo_clip) * reward
            output = -torch.min(surr1, surr2) * mask
        else:
            output = - input * reward * mask

        output = torch.sum(output) / batch_size + entropy_reg * torch.sum(entropy_minus) / batch_size

        return output


# ---- mle criterion
class SoftAttPlusLTGCriterion(nn.Module):
    def __init__(self, opt):
        super(SoftAttPlusLTGCriterion, self).__init__()
        self.use_label_smoothing = opt.use_label_smoothing
        self.label_smoothing_epsilon = opt.label_smoothing_epsilon

    # top_true should be Variable containing LongTensor, top_true is also the ground truth of gv1 and gv2
    def forward(self, log_prob, target, mask, gv, top_true, ltg_weight, gv_l1_penality):
        # truncate to the same size, input: (80L, 16L, 9488L), target: (80, 17)
        batch_size = log_prob.size(0)
        target = target[:, :log_prob.size(1)]
        mask = mask[:, :log_prob.size(1)]
        log_prob = to_contiguous(log_prob).view(-1, log_prob.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - log_prob.gather(1, target) * mask
        output = torch.sum(output) / batch_size

        gv_loss = nn.MultiLabelMarginLoss()(gv, top_true)

        zero_tensor = Variable(torch.zeros(gv.size()))
        zero_tensor = zero_tensor.cuda()

        # gv_l1_loss = nn.L1Loss(size_average=False)(gv, zero_tensor)
        gv_l1_loss = nn.SmoothL1Loss(size_average=False)(gv, zero_tensor)
        print('loss: ' + str(output.data[0]) + ', guiding loss: ' + str(gv_loss.data[0]) + ', l1 loss: ' + str(gv_l1_loss.data[0]))

        output = output + gv_loss * ltg_weight + gv_l1_loss * gv_l1_penality
        return output


class ReviewNetEnsembleCriterion(nn.Module):
    def __init__(self, opt):
        super(ReviewNetEnsembleCriterion, self).__init__()
        self.use_label_smoothing = opt.use_label_smoothing
        self.label_smoothing_epsilon = opt.label_smoothing_epsilon
        self.use_cuda = opt.use_cuda

    # top_true should be Variable containing LongTensor
    def forward(self, log_prob, target, mask, top_pred, top_true, reason_weight):
        # truncate to the same size, input: (80L, 16L, 9488L), target: (80, 17)
        batch_size = log_prob.size(0)  # 50
        target = target[:, :log_prob.size(1)]
        mask = mask[:, :log_prob.size(1)]
        if self.use_label_smoothing:
            K = log_prob.size(2)
            step_length = log_prob.size(1)
            target_ = torch.unsqueeze(target, 2)
            one_hot = torch.FloatTensor(batch_size, step_length, K).zero_()
            if self.use_cuda:
                one_hot = one_hot.cuda()

            one_hot.scatter_(2, target_.data, 1.0)
            one_hot = one_hot * (1.0 - self.label_smoothing_epsilon) + self.label_smoothing_epsilon / K
            output = -torch.sum(log_prob * Variable(one_hot), 2) * mask
            output = torch.sum(output) / batch_size
        else:
            input = to_contiguous(log_prob).view(-1, log_prob.size(2))  # 800, 9488
            target = to_contiguous(target).view(-1, 1)  # 800, 1
            mask = to_contiguous(mask).view(-1, 1)  # 800, 1
            output = - input.gather(1, target) * mask
            # output = torch.sum(output) / torch.sum(mask)
            output = torch.sum(output) / batch_size

        discriminative_loss = []
        for i in range(len(top_pred)):
            discriminative_loss.append(nn.MultiLabelMarginLoss()(top_pred[i], top_true))

        output = output + sum(discriminative_loss) * reason_weight/len(top_pred)

        return output


# only for eval
class TVCriterion(nn.Module):
    def __init__(self, opt):
        super(TVCriterion, self).__init__()
        self.use_label_smoothing = opt.use_label_smoothing
        self.label_smoothing_epsilon = opt.label_smoothing_epsilon
        self.use_cuda = opt.use_cuda

    # top_true should be Variable containing LongTensor
    def forward(self, log_prob, target, mask, top_pred, top_true, reason_weight):
        if isinstance(top_pred, list):
            top_pred = top_pred[-1]

        output = nn.MultiLabelMarginLoss()(top_pred, top_true)
        return output


class ReviewNetCriterion(nn.Module):
    def __init__(self, opt):
        super(ReviewNetCriterion, self).__init__()
        self.use_label_smoothing = opt.use_label_smoothing
        self.label_smoothing_epsilon = opt.label_smoothing_epsilon
        self.use_cuda = opt.use_cuda

    # top_true should be Variable containing LongTensor
    def forward(self, log_prob, target, mask, top_pred, top_true, reason_weight):
        # truncate to the same size, input: (80L, 16L, 9488L), target: (80, 17)
        batch_size = log_prob.size(0)  # 50
        # print('batch_size in ReviewNetCriterion'  + str(batch_size))
        target = target[:, :log_prob.size(1)]
        mask = mask[:, :log_prob.size(1)]
        if self.use_label_smoothing:
            K = log_prob.size(2)
            step_length = log_prob.size(1)
            target_ = torch.unsqueeze(target, 2)
            one_hot = torch.FloatTensor(batch_size, step_length, K).zero_()
            if self.use_cuda:
                one_hot = one_hot.cuda()

            one_hot.scatter_(2, target_.data, 1.0)
            one_hot = one_hot * (1.0 - self.label_smoothing_epsilon) + self.label_smoothing_epsilon / K
            output = -torch.sum(log_prob * Variable(one_hot), 2) * mask
            output = torch.sum(output) / batch_size
        else:
            input = to_contiguous(log_prob).view(-1, log_prob.size(2))  # 800, 9488
            target = to_contiguous(target).view(-1, 1)  # 800, 1
            mask = to_contiguous(mask).view(-1, 1)  # 800, 1
            output = - input.gather(1, target) * mask
            # output = torch.sum(output) / torch.sum(mask)
            output = torch.sum(output) / batch_size

        discriminative_loss = nn.MultiLabelMarginLoss()(top_pred, top_true)
        output = output + discriminative_loss * reason_weight

        return output


class LanguageModelCriterion(nn.Module):
    def __init__(self, opt):
        super(LanguageModelCriterion, self).__init__()
        self.use_label_smoothing = opt.use_label_smoothing
        self.label_smoothing_epsilon = opt.label_smoothing_epsilon
        self.use_cuda = opt.use_cuda

    def forward(self, input, target, mask):
        # truncate to the same size, input: (80L, 16L, 9488L), target: (80, 17)
        batch_size = input.size(0)
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        if self.use_label_smoothing:
            K = input.size(2)
            step_length = input.size(1)
            target_ = torch.unsqueeze(target, 2)
            one_hot = torch.FloatTensor(batch_size, step_length, K).zero_()
            if self.use_cuda:
                one_hot = one_hot.cuda()

            one_hot.scatter_(2, target_.data, 1.0)
            one_hot = one_hot * (1.0 - self.label_smoothing_epsilon) + self.label_smoothing_epsilon / K
            output = -torch.sum(input * Variable(one_hot), 2) * mask
            output = torch.sum(output) / batch_size
        else:
            input = to_contiguous(input).view(-1, input.size(2))
            target = to_contiguous(target).view(-1, 1)
            mask = to_contiguous(mask).view(-1, 1)
            output = - input.gather(1, target) * mask
            # output = torch.sum(output) / torch.sum(mask)
            output = torch.sum(output) / batch_size

        return output


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
