# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from misc.LSTMSoftAttentionCore import LSTMSoftAttentionCore
from misc.LSTMSoftAttentionNoInputCore import LSTMSoftAttentionNoInputCore
from misc.MixtureOfSoftmax import MixtureOfSoftmax
# import misc.utils as utils

import opts
import numpy as np


class ReviewNetModel(nn.Module):
    def __init__(self, opt):
        super(ReviewNetModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.drop_prob_reason = opt.drop_prob_reason
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.num_review_steps = opt.num_review_steps
        self.att_num = opt.att_num
        self.att_feat_size = opt.att_feat_size
        self.top_words_count = opt.top_words_count
        self.att_hid_size = opt.att_hid_size
        self.ss_prob = 0.0  # Schedule sampling probability
        self.review_maxout = opt.review_maxout
        self.decoder_maxout = opt.maxout
        self.use_cuda = opt.use_cuda
        self.use_mos = opt.use_mos
        self.num_expert = opt.num_expert

        self.fc2h = nn.Linear(self.fc_feat_size, self.rnn_size)

        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)

        self.review_steps = nn.ModuleList(
            [LSTMSoftAttentionNoInputCore(self.rnn_size, self.att_feat_size, self.att_num, self.att_hid_size, self.drop_prob_reason, self.review_maxout)
             for i in range(self.num_review_steps)
             ]
        )
        self.reason_linear = nn.Linear(self.rnn_size, self.top_words_count)
        self.decoder = LSTMSoftAttentionCore(self.input_encoding_size,
                                             self.rnn_size,
                                             self.rnn_size,
                                             self.num_review_steps,
                                             self.att_hid_size,
                                             self.drop_prob_lm,
                                             self.decoder_maxout)

        if self.use_mos:
            self.mos = MixtureOfSoftmax(self.rnn_size, self.rnn_size, self.num_expert, self.vocab_size+1)
        # self.decoder.cuda()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.fc2h.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)
        self.reason_linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, fc_feats, att_feats, seq):
        batch_size = fc_feats.size(0)

        init_h = self.fc2h(fc_feats)
        init_h = init_h.unsqueeze(0)
        init_c = init_h.clone()
        state = (init_h, init_c)
        # state0 = (state[0].clone(), state[1].clone())

        thought = []
        reason_mat_list = []
        for i in range(self.num_review_steps):
            output, state = self.review_steps[i](att_feats, state)
            thought.append(output)
            reason_mat_list.append(self.reason_linear(output))

        if self.use_cuda:
            thought_vectors = torch.stack(thought).transpose(0, 1).cuda().contiguous()
            reason_mat = torch.stack(reason_mat_list).transpose(0, 1).cuda().contiguous()
        else:
            thought_vectors = torch.stack(thought).transpose(0, 1).contiguous()
            reason_mat = torch.stack(reason_mat_list).transpose(0, 1).contiguous()

        reason_pred, _ = torch.max(reason_mat, 1)
        reason_pred = reason_pred.squeeze()

        # state = state0
        outputs = []
        for i in range(seq.size(1)):
            if i >= 1 and self.ss_prob > 0.0:  # otherwise no need to sample
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    # prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    # it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    prob_prev = torch.exp(outputs[-1].data)  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq[:, i].data.sum() == 0:
                break
            xt = self.embed(it)
            output, state = self.decoder(xt, thought_vectors, state)
            if self.use_mos:
                output = torch.log(self.mos(output.squeeze(0)))
            else:
                output = F.log_softmax(self.logit(output.squeeze(0)))
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1).contiguous(), reason_pred
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1).contiguous()

    def get_thought_vectors(self, fc_feats, att_feats, state):
        thought = []
        reason_mat_list = []
        for i in range(self.num_review_steps):
            output, state = self.review_steps[i](att_feats, state)
            thought.append(output)
            reason_mat_list.append(self.reason_linear(output))

        if self.use_cuda:
            thought_vectors = torch.stack(thought).transpose(0, 1).cuda().contiguous()
            reason_mat = torch.stack(reason_mat_list).transpose(0, 1).cuda().contiguous()
        else:
            thought_vectors = torch.stack(thought).transpose(0, 1).contiguous()
            reason_mat = torch.stack(reason_mat_list).transpose(0, 1).contiguous()
        reason_pred, _ = torch.max(reason_mat, 1)
        reason_pred = reason_pred.squeeze()
        return thought_vectors, reason_pred, state

    def get_init_state(self, fc_feats):
        init_h = self.fc2h(fc_feats)
        init_h = init_h.unsqueeze(0)
        init_c = init_h.clone()
        state = (init_h, init_c)
        return state

    def one_time_step(self, xt, fc_feats, att_feats, state):
        # xt = self.embed(Variable(it, requires_grad=False))
        output, state = self.decoder(xt, att_feats, state)
        # logprobs = F.log_softmax(self.logit(output))
        if self.use_mos:
            logit = self.mos(output)
        else:
            logit = self.logit(output)

        return logit, state

    def sample_beam(self, fc_feats, att_feats, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)
        fc_feat_size = fc_feats.size(1)

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

        for k in range(batch_size):
            init_h = self.fc2h(fc_feats[k].unsqueeze(0).expand(beam_size, fc_feat_size))
            init_h = init_h.unsqueeze(0)
            init_c = init_h.clone()
            state = (init_h, init_c)
            att_feats_current = att_feats[k].unsqueeze(0).expand(beam_size, att_feats.size(1), att_feats.size(2))
            att_feats_current = att_feats_current.contiguous()

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam

            thought_vectors = Variable(torch.zeros(beam_size, self.num_review_steps, self.rnn_size))
            if self.use_cuda:
                thought_vectors = thought_vectors.cuda()

            for i in range(self.num_review_steps):
                output, state = self.review_steps[i](att_feats_current,
                                                     state)
                thought_vectors[:, i, :] = output

            for t in range(self.seq_length + 1):
                if t == 0:  # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
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

                output, state = self.decoder(xt, thought_vectors, state)
                if self.use_mos:
                    logprobs = torch.log(self.mos(output))
                else:
                    logprobs = F.log_softmax(self.logit(output))

                # logprobs = F.log_softmax(self.logit(output))

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
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1), top_seq, top_prob

    def sample(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats.size(0)
        seq = []
        seqLogprobs = []
        logprobs_all = []

        init_h = self.fc2h(fc_feats)
        init_h = init_h.unsqueeze(0)
        init_c = init_h.clone()
        state = (init_h, init_c)

        thought = []
        reason_mat_list = []
        for i in range(self.num_review_steps):
            output, state = self.review_steps[i](att_feats, state)
            thought.append(output)
            reason_mat_list.append(self.reason_linear(output))

        if self.use_cuda:
            thought_vectors = torch.stack(thought).transpose(0, 1).cuda().contiguous()
            reason_mat = torch.stack(reason_mat_list).transpose(0, 1).cuda().contiguous()
        else:
            thought_vectors = torch.stack(thought).transpose(0, 1).contiguous()
            reason_mat = torch.stack(reason_mat_list).transpose(0, 1).contiguous()

        reason_pred, _ = torch.max(reason_mat, 1)
        reason_pred = reason_pred.squeeze()

        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_()
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
                sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False))  # gather the logprobs at sampled positions
                # sampleLogprobs = sampleLogprobs.data
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

            output, state = self.decoder(xt, thought_vectors, state)
            # logprobs = F.log_softmax(self.logit(output))
            if self.use_mos:
                logprobs = torch.log(self.mos(output))
            else:
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

    model = ReviewNetModel(opt)
    if opt.use_cuda:
        model = model.cuda()
    for idx, m in enumerate(model.modules()):
        print(idx, '->', m)
    all_p = list(model.parameters())
    print('number of parameter groups: ' + str(len(all_p)))

    fc_feats = Variable(torch.randn(opt.batch_size, opt.fc_feat_size))
    att_feats = Variable(torch.randn(opt.batch_size, opt.att_num, opt.att_feat_size))
    random_seq = torch.LongTensor(np.random.randint(100, size=(opt.batch_size, opt.seq_length)))
    seq = Variable(random_seq)

    if opt.use_cuda:
        fc_feats = fc_feats.cuda()
        att_feats = att_feats.cuda()
        seq = seq.cuda()

    out = model(fc_feats, att_feats, seq)
    sampled_seq, sampled_prob = model.sample(fc_feats, att_feats,
                                             {'beam_size': 1})

    sampled_seq, sampled_prob, top_seq, top_prob = model.sample(fc_feats, att_feats,
                                                                {'beam_size': 3})
    print(sampled_seq.size())
    print('passed!')
