# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from misc.LSTMSoftMultiAttentionFeatArrayNoInputCore import LSTMSoftMultiAttentionFeatArrayNoInputCore
from misc.LSTMSoftAttentionCore import LSTMSoftAttentionCore
from misc.AttentionModelCore import AttentionModelCore

import opts
import numpy as np


# caption_model: recurrent_fusion_model
# feature_type: feat_array

class LSTMFusionNoInputCore(nn.Module):
    def __init__(self, H_size, rnn_size, att_feat_size, att_num, att_hid_size, drop_prob_fusion, maxout=0):
        super(LSTMFusionNoInputCore, self).__init__()

        self.drop_prob_fusion = drop_prob_fusion
        self.att_hid_size = att_hid_size
        self.maxout = maxout
        self.H_size = H_size
        self.rnn_size = rnn_size
        self.att_feat_size = att_feat_size
        self.att_num = att_num

        self.att_model = AttentionModelCore(self.rnn_size, self.att_feat_size, self.att_num, self.att_hid_size)

        # Build a LSTM
        if self.maxout:
            self.H2h = nn.Linear(self.H_size, 5 * self.rnn_size)
            self.z2h = nn.Linear(self.att_feat_size, 5 * self.rnn_size)
        else:
            self.H2h = nn.Linear(self.H_size, 4 * self.rnn_size)
            self.z2h = nn.Linear(self.att_feat_size, 4 * self.rnn_size)

        self.dropout = nn.Dropout(self.drop_prob_fusion)

        # init
        initrange = 0.1
        self.H2h.weight.data.uniform_(-initrange, initrange)
        self.z2h.weight.data.uniform_(-initrange, initrange)

    def forward(self, H, att_feat, state):  # state = (pre_h, pre_c)
        pre_h = state[0][-1]
        pre_c = state[1][-1]

        z = self.att_model(pre_h, att_feat)

        all_input_sums = self.H2h(H) + self.z2h(z)

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


class FeatArrayFusionNoInputCore(nn.Module):
    def __init__(self, num_feat_array, rnn_size, att_feat_size, att_num, att_hid_size, drop_prob_fusion, maxout=0):
        # att_feat_size and att_num are list
        super(FeatArrayFusionNoInputCore, self).__init__()

        # self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = rnn_size
        self.drop_prob_fusion = drop_prob_fusion  #
        self.att_feat_size = att_feat_size
        self.att_num = att_num
        self.att_hid_size = att_hid_size
        self.maxout = maxout
        self.num_feat_array = num_feat_array
        self.H_size = num_feat_array * rnn_size
        self.Z_size = sum(att_feat_size)

        self.lstm = nn.ModuleList(
            [LSTMFusionNoInputCore(self.H_size, self.rnn_size, self.att_feat_size[i], self.att_num[i],
                                   self.att_hid_size, self.drop_prob_fusion)
             for i in range(self.num_feat_array)]
        )

        self.dropout = nn.Dropout(self.drop_prob_fusion)

    def forward(self, att_seq, state_list):  # state is a list
        H_list = []
        for i in range(self.num_feat_array):
            pre_h = state_list[i][0][-1]
            H_list.append(pre_h)

        H = torch.cat(H_list, 1)

        output_list = []
        for i in range(self.num_feat_array):
            output, state_list[i] = self.lstm[i](H, att_seq[i], state_list[i])
            output_list.append(output)

        return output_list, state_list


class RecurrentFusionModel(nn.Module):
    def __init__(self, opt):
        super(RecurrentFusionModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_reason = opt.drop_prob_reason
        self.drop_prob_fusion = opt.drop_prob_fusion
        self.seq_length = opt.seq_length
        # self.fc_feat_size = opt.fc_feat_size
        self.num_review_steps = opt.num_review_steps
        self.num_review_steps_0 = opt.num_review_steps_0
        # self.att_num = opt.att_num
        # self.att_feat_size = opt.att_feat_size
        self.top_words_count = opt.top_words_count
        self.att_hid_size = opt.att_hid_size
        self.ss_prob = 0.0  # Schedule sampling probability
        self.review_maxout = opt.review_maxout
        self.decoder_maxout = opt.maxout
        self.fusion_maxout = opt.fusion_maxout
        self.use_cuda = opt.use_cuda

        self.feat_array_info = opt.feat_array_info
        self.num_feat_array = len(self.feat_array_info)

        self.fc_feat_size = []
        self.att_feat_size = []
        self.att_num = []
        for j in range(self.num_feat_array):
            self.fc_feat_size.append(self.feat_array_info[j]['fc_feat_size'])
            self.att_feat_size.append(self.feat_array_info[j]['att_feat_size'])
            self.att_num.append(self.feat_array_info[j]['att_num'])

        self.fc2h = nn.ModuleList([nn.Linear(self.fc_feat_size[i], self.rnn_size) for i in range(self.num_feat_array)])

        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)

        self.review_steps_individual = nn.ModuleList([
            FeatArrayFusionNoInputCore(self.num_feat_array, self.rnn_size, self.att_feat_size, self.att_num,
                                       self.att_hid_size, self.drop_prob_fusion, self.fusion_maxout)
            for _ in range(self.num_review_steps_0)
        ])
        self.reason_linear_individual = nn.ModuleList([
            nn.Linear(self.rnn_size, self.top_words_count) for _ in range(self.num_feat_array)
        ])

        self.review_steps = nn.ModuleList([
            LSTMSoftMultiAttentionFeatArrayNoInputCore(self.rnn_size,
                                                       [self.rnn_size] * self.num_feat_array,
                                                       [self.num_review_steps_0] * self.num_feat_array,
                                                       self.att_hid_size,
                                                       self.drop_prob_reason,
                                                       self.review_maxout)
            for _ in range(self.num_review_steps)
        ])
        #
        self.reason_linear = nn.Linear(self.rnn_size, self.top_words_count)
        self.decoder = LSTMSoftAttentionCore(self.input_encoding_size,
                                             self.rnn_size,
                                             self.rnn_size,
                                             self.num_review_steps,
                                             self.att_hid_size,
                                             self.drop_prob_lm,
                                             self.decoder_maxout)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.reason_linear.weight.data.uniform_(-initrange, initrange)
        for i in range(self.num_feat_array):
            self.reason_linear_individual[i].weight.data.uniform_(-initrange, initrange)
            self.fc2h[i].weight.data.uniform_(-initrange, initrange)

    def forward(self, fc_feats, att_feats, seq):
        batch_size = fc_feats[0].size(0)
        state_list = []
        # H_list = []
        for i in range(self.num_feat_array):
            init_h = self.fc2h[i](fc_feats[i])
            # H_list.append(init_h)
            init_h = init_h.unsqueeze(0)
            init_c = init_h.clone()
            state = (init_h, init_c)
            state_list.append(state)

        thought_vectors_list = [[] for i in range(self.num_feat_array)]
        reason_mat_list = [[] for i in range(self.num_feat_array)]

        for i in range(self.num_review_steps_0):
            output_list, state_list = self.review_steps_individual[i](att_feats, state_list)
            for j in range(self.num_feat_array):
                thought_vectors_list[j].append(output_list[j])
                reason_mat_list[j].append(self.reason_linear_individual[j](output_list[j]))

        thought_vectors = []
        reason_pred = []
        for i in range(self.num_feat_array):
            if self.use_cuda:
                thought_vectors_temp = torch.stack(thought_vectors_list[i]).transpose(0, 1).cuda().contiguous()
                reason_mat_temp = torch.stack(reason_mat_list[i]).transpose(0, 1).cuda().contiguous()
            else:
                thought_vectors_temp = torch.stack(thought_vectors_list[i]).transpose(0, 1).contiguous()
                reason_mat_temp = torch.stack(reason_mat_list[i]).transpose(0, 1).contiguous()

            reason_pred_temp, _ = torch.max(reason_mat_temp, 1)
            reason_pred.append(reason_pred_temp.squeeze())
            thought_vectors.append(thought_vectors_temp)

        init_h_review, init_c_review = map(sum, zip(*state_list))
        init_h_review = init_h_review / self.num_feat_array
        init_c_review = init_c_review / self.num_feat_array

        state_review = (init_h_review, init_c_review)

        thought_comb = []
        reason_mat_list_comb = []
        for i in range(self.num_review_steps):
            output, state_review = self.review_steps[i](thought_vectors, state_review)
            thought_comb.append(output)
            reason_mat_list_comb.append(self.reason_linear(output))

        if self.use_cuda:
            thought_vectors_comb = torch.stack(thought_comb).transpose(0, 1).cuda().contiguous()
            reason_mat_comb = torch.stack(reason_mat_list_comb).transpose(0, 1).cuda().contiguous()
        else:
            thought_vectors_comb = torch.stack(thought_comb).transpose(0, 1).contiguous()
            reason_mat_comb = torch.stack(reason_mat_list_comb).transpose(0, 1).contiguous()

        reason_pred_comb, _ = torch.max(reason_mat_comb, 1)
        reason_pred_comb = reason_pred_comb.squeeze()
        reason_pred.append(reason_pred_comb)

        state_decode = state_review
        outputs = []
        for i in range(seq.size(1)):
            if i >= 1 and self.ss_prob > 0.0:  # otherwise no need to sample
                sample_prob = fc_feats[0].data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[-1].data)  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq[:, i].data.sum() == 0:
                break
            xt = self.embed(it)
            output, state_decode = self.decoder(xt, thought_vectors_comb, state_decode)
            output = F.log_softmax(self.logit(output.squeeze(0)))
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1).contiguous(), reason_pred

    def get_thought_vectors(self, fc_feats, att_feats, state_list):
        thought_vectors_list = [[] for i in range(self.num_feat_array)]
        reason_mat_list = [[] for i in range(self.num_feat_array)]

        for i in range(self.num_review_steps_0):
            output_list, state_list = self.review_steps_individual[i](att_feats, state_list)
            for j in range(self.num_feat_array):
                thought_vectors_list[j].append(output_list[j])
                reason_mat_list[j].append(self.reason_linear_individual[j](output_list[j]))

        thought_vectors = []
        reason_pred = []
        for i in range(self.num_feat_array):
            if self.use_cuda:
                thought_vectors_temp = torch.stack(thought_vectors_list[i]).transpose(0, 1).cuda().contiguous()
                reason_mat_temp = torch.stack(reason_mat_list[i]).transpose(0, 1).cuda().contiguous()
            else:
                thought_vectors_temp = torch.stack(thought_vectors_list[i]).transpose(0, 1).contiguous()
                reason_mat_temp = torch.stack(reason_mat_list[i]).transpose(0, 1).contiguous()

            reason_pred_temp, _ = torch.max(reason_mat_temp, 1)
            reason_pred.append(reason_pred_temp.squeeze())
            thought_vectors.append(thought_vectors_temp)

        init_h_review, init_c_review = map(sum, zip(*state_list))
        init_h_review = init_h_review / self.num_feat_array
        init_c_review = init_c_review / self.num_feat_array

        state_review = (init_h_review, init_c_review)

        thought_comb = []
        reason_mat_list_comb = []
        for i in range(self.num_review_steps):
            output, state_review = self.review_steps[i](thought_vectors, state_review)
            thought_comb.append(output)
            reason_mat_list_comb.append(self.reason_linear(output))

        if self.use_cuda:
            thought_vectors_comb = torch.stack(thought_comb).transpose(0, 1).cuda().contiguous()
            reason_mat_comb = torch.stack(reason_mat_list_comb).transpose(0, 1).cuda().contiguous()
        else:
            thought_vectors_comb = torch.stack(thought_comb).transpose(0, 1).contiguous()
            reason_mat_comb = torch.stack(reason_mat_list_comb).transpose(0, 1).contiguous()

        reason_pred_comb, _ = torch.max(reason_mat_comb, 1)
        reason_pred_comb = reason_pred_comb.squeeze()
        reason_pred.append(reason_pred_comb)

        return thought_vectors_comb, reason_pred, state_review

    def get_init_state(self, fc_feats):
        state_list = []
        # H_list = []
        for i in range(self.num_feat_array):
            init_h = self.fc2h[i](fc_feats[i])
            # H_list.append(init_h)
            init_h = init_h.unsqueeze(0)
            init_c = init_h.clone()
            state = (init_h, init_c)
            state_list.append(state)
        return state_list

    def one_time_step(self, xt, fc_feats, thought_vectors_comb, state_decode):
        # xt = self.embed(Variable(it, requires_grad=False))
        output, state_decode = self.decoder(xt, thought_vectors_comb, state_decode)
        # logprobs = F.log_softmax(self.logit(output))
        logit = self.logit(output)
        return logit, state_decode

    def sample_beam(self, fc_feats, att_feats, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats[0].size(0)

        fc_feat_size_list = []
        for fc in fc_feats:
            fc_feat_size_list.append(fc.size(1))

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        if self.use_cuda:
            seq = seq.cuda()
            seqLogprobs = seqLogprobs.cuda()

        top_seq = []
        top_prob = [[] for _ in range(batch_size)]

        self.done_beams = [[] for _ in range(batch_size)]

        reason_pred_batch = []
        for k in range(batch_size):
            fc_feats_current = []
            att_feats_current = []
            for feat_id in range(self.num_feat_array):
                fc_feats_current.append(fc_feats[feat_id][k].unsqueeze(0).expand(beam_size, fc_feat_size_list[feat_id]))
                att_feats_current.append(
                    att_feats[feat_id][k].unsqueeze(0).expand(beam_size, att_feats[feat_id].size(1),
                                                              att_feats[feat_id].size(2)))

            for feat_id in range(self.num_feat_array):
                fc_feats_current[feat_id] = fc_feats_current[feat_id].contiguous()
                att_feats_current[feat_id] = att_feats_current[feat_id].contiguous()

            state_current_list = []
            for feat_id in range(self.num_feat_array):
                init_h = self.fc2h[feat_id](fc_feats_current[feat_id])
                init_h = init_h.unsqueeze(0)
                init_c = init_h.clone()
                state_current = (init_h, init_c)
                state_current_list.append(state_current)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam

            thought_vectors_list = [[] for i in range(self.num_feat_array)]
            reason_mat_list = [[] for i in range(self.num_feat_array)]

            for i in range(self.num_review_steps_0):
                output_list, state_current_list = self.review_steps_individual[i](att_feats_current,
                                                                                  state_current_list)
                for j in range(self.num_feat_array):
                    thought_vectors_list[j].append(output_list[j])
                    reason_mat_list[j].append(self.reason_linear_individual[j](output_list[j]))

            thought_vectors = []
            reason_pred = []
            for i in range(self.num_feat_array):
                if self.use_cuda:
                    thought_vectors_temp = torch.stack(thought_vectors_list[i]).transpose(0, 1).cuda().contiguous()
                    reason_mat_temp = torch.stack(reason_mat_list[i]).transpose(0, 1).cuda().contiguous()
                else:
                    thought_vectors_temp = torch.stack(thought_vectors_list[i]).transpose(0, 1).contiguous()
                    reason_mat_temp = torch.stack(reason_mat_list[i]).transpose(0, 1).contiguous()

                reason_pred_temp, _ = torch.max(reason_mat_temp, 1)
                reason_pred.append(reason_pred_temp.squeeze())
                thought_vectors.append(thought_vectors_temp)

            init_h_review, init_c_review = map(sum, zip(*state_current_list))
            init_h_review = init_h_review / self.num_feat_array
            init_c_review = init_c_review / self.num_feat_array

            state_review = (init_h_review, init_c_review)

            thought_comb = []
            reason_mat_list_comb = []
            for i in range(self.num_review_steps):
                output, state_review = self.review_steps[i](thought_vectors, state_review)
                thought_comb.append(output)
                reason_mat_list_comb.append(self.reason_linear(output))

            if self.use_cuda:
                thought_vectors_comb = torch.stack(thought_comb).transpose(0, 1).cuda().contiguous()
                reason_mat_comb = torch.stack(reason_mat_list_comb).transpose(0, 1).cuda().contiguous()
            else:
                thought_vectors_comb = torch.stack(thought_comb).transpose(0, 1).contiguous()
                reason_mat_comb = torch.stack(reason_mat_list_comb).transpose(0, 1).contiguous()

            reason_pred_comb, _ = torch.max(reason_mat_comb, 1)
            reason_pred_comb = reason_pred_comb.squeeze()
            reason_pred.append(reason_pred_comb)
            reason_pred_batch.append(reason_pred)

            state = state_review

            for t in range(self.seq_length + 1):
                if t == 0:  # input <bos>
                    it = fc_feats[0].data.new(beam_size).long().zero_()
                    xt = self.embed(Variable(it, requires_grad=False))
                    # xt = self.img_embed(fc_feats[k:k+1]).expand(beam_size, self.input_encoding_size)
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.float()  # lets go to CPU for more efficiency in indexing operations
                    # ys: beam_size * (Vab_szie + 1)
                    ys, ix = torch.sort(logprobsf, 1,
                                        True)  # sorted array of logprobs along each previous beam (last true = descending)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if t == 1:  # at first time step only the first beam is active
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            # compute logprob of expanding beam q with word in (sorted) position c
                            local_logprob = ys[q, c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            if t > 1 and beam_seq[t - 2, q] == 0:
                                continue
                            candidates.append({'c': ix.data[q, c], 'q': q, 'p': candidate_logprob.data[0],
                                               'r': local_logprob.data[0]})

                    if len(candidates) == 0:
                        break
                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # construct new beams
                    new_state = [_.clone() for _ in state]
                    if t > 1:
                        # well need these as reference when we fork beams around
                        beam_seq_prev = beam_seq[:t - 1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t - 1].clone()

                    for vix in range(min(beam_size, len(candidates))):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if t > 1:
                            beam_seq[:t - 1, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t - 1, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        for state_ix in range(len(new_state)):
                            # copy over state in previous beam q to new beam at vix
                            new_state[state_ix][0, vix] = state[state_ix][0, v['q']]  # dimension one is time step

                        # append new end terminal at the end of this beam
                        beam_seq[t - 1, vix] = v['c']  # c'th word is the continuation
                        beam_seq_logprobs[t - 1, vix] = v['r']  # the raw logprob here
                        beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam

                        if v['c'] == 0 or t == self.seq_length:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                       'logps': beam_seq_logprobs[:, vix].clone(),
                                                       'p': beam_logprobs_sum[vix]
                                                       })

                    # encode as vectors
                    it = beam_seq[t - 1]
                    if self.use_cuda:
                        xt = self.embed(Variable(it.cuda()))
                    else:
                        xt = self.embed(Variable(it))

                if t >= 1:
                    state = new_state

                output, state = self.decoder(xt, thought_vectors_comb, state)
                logprobs = F.log_softmax(self.logit(output))

            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']

            # save result
            l = len(self.done_beams[k])
            top_seq_cur = torch.LongTensor(l, self.seq_length).zero_()

            for temp_index in range(l):
                top_seq_cur[temp_index] = self.done_beams[k][temp_index]['seq'].clone()
                top_prob[k].append(self.done_beams[k][temp_index]['p'])

            top_seq.append(top_seq_cur)
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1), top_seq, top_prob, reason_pred_batch

    def sample(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats[0].size(0)
        seq = []
        seqLogprobs = []
        logprobs_all = []

        state_list = []
        # H_list = []
        for i in range(self.num_feat_array):
            init_h = self.fc2h[i](fc_feats[i])
            # H_list.append(init_h)
            init_h = init_h.unsqueeze(0)
            init_c = init_h.clone()
            state = (init_h, init_c)
            state_list.append(state)

        thought_vectors_list = [[] for i in range(self.num_feat_array)]
        reason_mat_list = [[] for i in range(self.num_feat_array)]

        for i in range(self.num_review_steps_0):
            output_list, state_list = self.review_steps_individual[i](att_feats, state_list)
            for j in range(self.num_feat_array):
                thought_vectors_list[j].append(output_list[j])
                reason_mat_list[j].append(self.reason_linear_individual[j](output_list[j]))

        thought_vectors = []
        reason_pred = []
        for i in range(self.num_feat_array):
            if self.use_cuda:
                thought_vectors_temp = torch.stack(thought_vectors_list[i]).transpose(0, 1).cuda().contiguous()
                reason_mat_temp = torch.stack(reason_mat_list[i]).transpose(0, 1).cuda().contiguous()
            else:
                thought_vectors_temp = torch.stack(thought_vectors_list[i]).transpose(0, 1).contiguous()
                reason_mat_temp = torch.stack(reason_mat_list[i]).transpose(0, 1).contiguous()

            reason_pred_temp, _ = torch.max(reason_mat_temp, 1)
            reason_pred.append(reason_pred_temp.squeeze())
            thought_vectors.append(thought_vectors_temp)

        init_h_review, init_c_review = map(sum, zip(*state_list))
        init_h_review = init_h_review / self.num_feat_array
        init_c_review = init_c_review / self.num_feat_array

        state_review = (init_h_review, init_c_review)

        thought_comb = []
        reason_mat_list_comb = []
        for i in range(self.num_review_steps):
            output, state_review = self.review_steps[i](thought_vectors, state_review)
            thought_comb.append(output)
            reason_mat_list_comb.append(self.reason_linear(output))

        if self.use_cuda:
            thought_vectors_comb = torch.stack(thought_comb).transpose(0, 1).cuda().contiguous()
            reason_mat_comb = torch.stack(reason_mat_list_comb).transpose(0, 1).cuda().contiguous()
        else:
            thought_vectors_comb = torch.stack(thought_comb).transpose(0, 1).contiguous()
            reason_mat_comb = torch.stack(reason_mat_list_comb).transpose(0, 1).contiguous()

        reason_pred_comb, _ = torch.max(reason_mat_comb, 1)
        reason_pred_comb = reason_pred_comb.squeeze()
        reason_pred.append(reason_pred_comb)

        state_decode = state_review

        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats[0].data.new(batch_size).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu()  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                if self.use_cuda:
                    it = torch.multinomial(prob_prev, 1).cuda()
                else:
                    it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, Variable(it,
                                                             requires_grad=False))  # gather the logprobs at sampled positions
                # sampleLogprobs = sampleLogprobs.data()
                it = it.view(-1).long()  # and flatten indices for downstream processing

            xt = self.embed(Variable(it, requires_grad=False))

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it)  # seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))

            output, state_decode = self.decoder(xt, thought_vectors_comb, state_decode)
            logprobs = F.log_softmax(self.logit(output))
            logprobs_all.append(logprobs)

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), \
               torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1), \
               torch.cat([_.unsqueeze(1) for _ in logprobs_all], 1).contiguous(), \
               reason_pred


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt.vocab_size = 2000
    opt.seq_length = 30
    opt.sample_max = 1
    opt.use_cuda = 0

    opt.fc_feat_size_1 = 2048
    opt.fc_feat_size_2 = 1536
    opt.fc_feat_size_3 = 2048
    opt.fc_feat_size_4 = 2208

    opt.att_num_1 = 196
    opt.att_num_2 = 64
    opt.att_num_3 = 64
    opt.att_num_4 = 49

    opt.att_feat_size_1 = 2048
    opt.att_feat_size_2 = 1536
    opt.att_feat_size_3 = 1280
    opt.att_feat_size_4 = 2208

    model = RecurrentFusionModel(opt)

    fc_feats_1 = Variable(torch.randn(opt.batch_size, opt.fc_feat_size_1))
    fc_feats_2 = Variable(torch.randn(opt.batch_size, opt.fc_feat_size_2))
    fc_feats_3 = Variable(torch.randn(opt.batch_size, opt.fc_feat_size_3))
    fc_feats_4 = Variable(torch.randn(opt.batch_size, opt.fc_feat_size_4))

    att_feats_1 = Variable(torch.randn(opt.batch_size, opt.att_num_1, opt.att_feat_size_1))
    att_feats_2 = Variable(torch.randn(opt.batch_size, opt.att_num_2, opt.att_feat_size_2))
    att_feats_3 = Variable(torch.randn(opt.batch_size, opt.att_num_3, opt.att_feat_size_3))
    att_feats_4 = Variable(torch.randn(opt.batch_size, opt.att_num_4, opt.att_feat_size_4))

    random_seq = torch.LongTensor(np.random.randint(100, size=(opt.batch_size, opt.seq_length)))
    seq = Variable(random_seq)

    if opt.use_cuda:
        model = model.cuda()
        fc_feats_1 = fc_feats_1.cuda()
        fc_feats_2 = fc_feats_2.cuda()
        fc_feats_3 = fc_feats_3.cuda()
        fc_feats_4 = fc_feats_4.cuda()
        att_feats_1 = att_feats_1.cuda()
        att_feats_2 = att_feats_2.cuda()
        att_feats_3 = att_feats_3.cuda()
        att_feats_4 = att_feats_4.cuda()
        seq = seq.cuda()

    fc_feats = [fc_feats_1, fc_feats_2, fc_feats_3, fc_feats_4]
    att_feats = [att_feats_1, att_feats_2, att_feats_3, att_feats_4]

    out = model(fc_feats, att_feats, seq)
    sampled_seq, sampled_prob, logprobs_all, reason_pred = model.sample(fc_feats, att_feats,
                                                                        {'beam_size': 1})
    print(sampled_seq.size())
