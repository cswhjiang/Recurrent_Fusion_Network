# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
import operator
import torch.nn.functional as F
from random import randint


def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    r = randint(0, 100000)
    model_id = model_id + '_' + str(r)

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


# greedy search or sample
def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    val_images_use = eval_kwargs.get('val_images_use', -1)
    split = eval_kwargs.get('eval_split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 1)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    caption_model = eval_kwargs.get('caption_model', 'fc')
    reason_weight = eval_kwargs.get('reason_weight', 10)
    guiding_weight = eval_kwargs.get('guiding_weight', 10)
    guiding_l1_penality = eval_kwargs.get('guiding_l1_penality', 0.00001)
    use_cuda = eval_kwargs.get('use_cuda', 0)
    feature_type = eval_kwargs.get('feature_type', 'resnet')
    rank = eval_kwargs.get('rank', 0)
    sample_max = eval_kwargs.get('sample_max', '1')
    print_beam_candidate = eval_kwargs.get('print_beam_candidate', 0)
    print_top_words = eval_kwargs.get('print_top_words', 0)
    # print(eval_kwargs)
    print(
        '---------- eval split: ' + split + ', num: ' + str(val_images_use) + ', rank ' + str(rank) + ' ------------ ')

    print('beam_size: ' + str(beam_size))
    # assert beam_size == 1, "beam_size should be 1"

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 0
    log_probs_sentence_sum = 0
    predictions = []

    thought_vectors_review_block_list = []
    discriminative_loss_list = []

    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if feature_type == 'feat_array':
            fc_feat_array_temp = data['fc_feats_array']
            att_feat_array_temp = data['att_feats_array']
            num_feat_array = len(fc_feat_array_temp)
            fc_feat_array = [[] for _ in range(num_feat_array)]
            att_feat_array = [[] for _ in range(num_feat_array)]
            for feat_id in range(len(fc_feat_array)):
                if use_cuda:
                    fc_feat_array[feat_id] = Variable(torch.from_numpy(fc_feat_array_temp[feat_id]),
                                                      volatile=True).cuda()
                    att_feat_array[feat_id] = Variable(torch.from_numpy(att_feat_array_temp[feat_id]),
                                                       volatile=True).cuda()
                else:
                    fc_feat_array[feat_id] = Variable(torch.from_numpy(fc_feat_array_temp[feat_id]), volatile=True)
                    att_feat_array[feat_id] = Variable(torch.from_numpy(att_feat_array_temp[feat_id]), volatile=True)

            tmp = [data['labels'], data['masks'], data['top_words']]
            if use_cuda:
                tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            else:
                tmp = [Variable(torch.from_numpy(_), volatile=True) for _ in tmp]
            labels, masks, top_words = tmp

        else:
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['top_words']]
            if use_cuda:
                tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            else:
                tmp = [Variable(torch.from_numpy(_), volatile=True) for _ in tmp]
            fc_feats, att_feats, labels, masks, top_words = tmp

        if caption_model == 'show_tell':
            log_prob = model(fc_feats, att_feats, labels)  # (80L, 16L, 9488L)
            loss = crit(log_prob, labels[:, 1:], masks[:, 1:]).data[0]

        elif caption_model == 'review_net':
            log_prob, top_pred = model(fc_feats, att_feats,labels)  # (80L, 16L, 9488L)
            loss = crit(log_prob, labels[:, 1:], masks[:, 1:], top_pred, top_words, reason_weight)
            loss = loss.data[0]
            # discriminative_loss_list.append(discriminative_loss)

        elif caption_model == 'recurrent_fusion_model':
            log_prob, top_pred = model(fc_feat_array, att_feat_array, labels)  # (80L, 16L, 9488L)
            loss = crit(log_prob, labels[:, 1:], masks[:, 1:], top_pred, top_words, reason_weight).data[0]
        else:
            raise Exception("caption_model not supported: {}".format(caption_model))
        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        if split == 'val' or split == 'test':
            if feature_type == 'feat_array':
                fc_feat_array_temp = data['fc_feats_array']
                att_feat_array_temp = data['att_feats_array']
                num_feat_array = len(fc_feat_array_temp)

                fc_feat_array = [[] for _ in range(num_feat_array)]
                att_feat_array = [[] for _ in range(num_feat_array)]
                for feat_id in range(num_feat_array):
                    fc_temp = fc_feat_array_temp[feat_id][np.arange(loader.batch_size) * loader.seq_per_img]
                    att_temp = att_feat_array_temp[feat_id][np.arange(loader.batch_size) * loader.seq_per_img]
                    if use_cuda:
                        fc_feat_array[feat_id] = Variable(torch.from_numpy(fc_temp), volatile=True).cuda()
                        att_feat_array[feat_id] = Variable(torch.from_numpy(att_temp), volatile=True).cuda()
                    else:
                        fc_feat_array[feat_id] = Variable(torch.from_numpy(fc_temp), volatile=True)
                        att_feat_array[feat_id] = Variable(torch.from_numpy(att_temp), volatile=True)

                tmp = [data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
                       data['masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                       data['top_words'][np.arange(loader.batch_size) * loader.seq_per_img]
                       ]
                # obj_feats, mil_feats, matching_feats, labels, masks, top_words = tmp

            else:
                tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                       data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                       ]

            if use_cuda:
                tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            else:
                tmp = [Variable(torch.from_numpy(_), volatile=True) for _ in tmp]

            if feature_type == 'feat_array':
                labels, masks, top_words = tmp
                sample_result_tupple = model.sample(fc_feat_array, att_feat_array,
                                                    {'beam_size': beam_size, 'sample_max': sample_max})
            else:
                fc_feats, att_feats = tmp
                sample_result_tupple = model.sample(fc_feats, att_feats,
                                                    {'beam_size': beam_size, 'sample_max': sample_max})

            seq = sample_result_tupple[0]  # torch.cuda.LongTensor
            if len(sample_result_tupple) == 5:
                thought_vectors_review_block_list.append(sample_result_tupple[4])

            seqLogprobs = sample_result_tupple[1]  # torch.cuda.FloatTensor
            log_probs_sentence = torch.sum(seqLogprobs * (seq > 0).type(torch.cuda.FloatTensor), 1)
            log_probs_sentence_sum = log_probs_sentence_sum + torch.mean(log_probs_sentence)

            if print_top_words and beam_size > 1:
                reason_pred_batch = sample_result_tupple[4]

        # if beam_size == 1:
        #     seq, _, = model.sample(fc_feats, att_feats, mil_feats, matching_feats, {'beam_size': beam_size})
        # else:
        #     seq, _, _, _ = model.sample(fc_feats, att_feats, mil_feats, matching_feats, {'beam_size': beam_size})

        # set_trace()
        if split == 'val' or split == 'test':
            sents = utils.decode_sequence(loader.get_vocab(), seq)

            for k, sent in enumerate(sents):
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                predictions.append(entry)
                if print_beam_candidate > 0:
                    print('%s\t%s' % (entry['image_id'], entry['caption']))
                if print_top_words and beam_size > 1:
                    reason_pred = reason_pred_batch[k]
                    for r_idnex, reason_pred_temp in enumerate(reason_pred):
                        _, top_index = torch.sort(reason_pred_temp[0], descending=True)
                        top_index = top_index.cpu().data.numpy()
                        top_index = top_index[: 10]
                        top_words = ""
                        for a in range(len(top_index)):
                            top_words = top_words + loader.top_words[top_index[a]] + ' '

                        print('%s_%s\t%s' % (entry['image_id'], str(r_idnex), top_words))
        else:
            print(n, loss_sum / loss_evals)

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if val_images_use != -1:
            ix1 = min(ix1, val_images_use)
        for i in range(n - ix1):
            if split == 'val' or split == 'test':
                predictions.pop()
        if verbose:
            print('evaluating validation preformance ... %d/%d (%f)' % (ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if n >= val_images_use:
            break

    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, 'eval_split_' + eval_kwargs['id'] + '_' + str(rank), split)
    else:
        lang_stats = None


    # Switch back to training mode
    model.train()
    print('log_probs_sentence_mean:' + str(log_probs_sentence_sum / loss_evals))
    return loss_sum / loss_evals, predictions, lang_stats


def model_ensemble_feat_array_one_step(model_list, xt_list, fc_feat_array, att_feat_array, mil_feats, matching_feats,
                                       state_list,
                                       thought_vector_list, caption_model):
    logit_list = []
    state_list_new = []
    n = len(model_list)
    for m in range(n):
        model = model_list[m]
        state = state_list[m]
        xt = xt_list[m]
        logit, state = model.one_time_step(xt, fc_feat_array, thought_vector_list[m], mil_feats, matching_feats, state)
        logit_list.append(logit)
        state_list_new.append(state)

    logit_mean = logit_list[0].clone()
    logit_mean.data.zero_()
    for l in logit_list:
        logit_mean = logit_mean + l

    logit_mean = logit_mean / n
    logprob = F.log_softmax(logit_mean)

    return logit_list, state_list_new, logprob


def model_ensemble_feat_array_one_step_multi_gpu(model_list, xt_list, fc_feat_array_multi_gpu, att_feat_array_multi_gpu,
                                                 mil_feats_multi_gpu, matching_feats_multi_gpu, state_list,
                                                 thought_vector_list, model_id_2_gpu_id):
    logit_list = []
    state_list_new = []
    n = len(model_list)
    for m in range(n):
        gpu_id = model_id_2_gpu_id[m]
        model = model_list[m]
        state = state_list[m]
        xt = xt_list[m]
        logit, state = model.one_time_step(xt.cuda(gpu_id), fc_feat_array_multi_gpu[gpu_id], thought_vector_list[m],
                                           mil_feats_multi_gpu[gpu_id], matching_feats_multi_gpu[gpu_id], state)
        logit_list.append(logit)
        state_list_new.append(state)

    logit_mean = logit_list[0].clone()
    logit_mean.data.zero_()
    for l in logit_list:
        logit_mean = logit_mean + l.cuda()

    logit_mean = logit_mean / n
    logprob = F.log_softmax(logit_mean)

    return logit_list, state_list_new, logprob


# one step
def model_ensemble_one_step(model_list, xt_list, fc_feats, att_feats, mil_feats, matching_feats, state_list,
                            thought_vector_list, gv_decoder_list, caption_model):
    logit_list = []
    state_list_new = []
    n = len(model_list)
    for m in range(n):
        model = model_list[m]
        state = state_list[m]
        xt = xt_list[m]

        if caption_model == 'fc' or caption_model == 'fc_mil' or caption_model == 'show_tell' \
                or caption_model == 'show_attend_tell' or caption_model == 'show_attend_tell_mil' \
                or caption_model == 'att2in' or caption_model == 'att2in_mil':
            logit, state = model.one_time_step(xt, fc_feats, att_feats, mil_feats, matching_feats, state)

        elif caption_model == 'guiding_net' or caption_model == 'guiding_net_2' or caption_model == 'guiding_net_plus_matching':
            logit, state = model.one_time_step(xt, fc_feats, thought_vector_list[m], gv_decoder_list[m], mil_feats,
                                               matching_feats, state)

        elif caption_model == 'show_attend_tell_ltg':
            logit, state = model.one_time_step(xt, fc_feats, att_feats, gv_decoder_list[m], matching_feats, state)

        elif caption_model == 'review_net' or caption_model == 'review_net_mil' or caption_model == 'multimodal':
            logit, state = model.one_time_step(xt, fc_feats, thought_vector_list[m], mil_feats, matching_feats, state)

        elif caption_model == 'review_blocks_32':
            logit, state = model.one_time_step(xt, fc_feats, thought_vector_list[m], mil_feats, matching_feats, state)
        else:
            raise Exception("caption_model not supported: {}".format(caption_model))

        logit_list.append(logit)
        state_list_new.append(state)

    logit_mean = logit_list[0].clone()
    logit_mean.data.zero_()
    for l in logit_list:
        logit_mean = logit_mean + l

    logit_mean = logit_mean / n
    logprob = F.log_softmax(logit_mean)

    return logit_list, state_list_new, logprob


def get_xt_list(model_list, it):
    xt_list = []
    for model in model_list:
        xt = model.embed(Variable(it, requires_grad=False))
        xt_list.append(xt)

    return xt_list


def get_xt_list_multi_gpu(model_list, it, model_id_2_gpu_id):
    xt_list = []
    for model_id, model in enumerate(model_list):
        gpu_id = model_id_2_gpu_id[model_id]
        with torch.cuda.device(gpu_id):
            it_var = Variable(it, requires_grad=False).cuda()
            xt = model.embed(it_var)
        xt_list.append(xt)

    return xt_list


# ensemble for beam search
def eval_ensemble(model_list, loader, eval_kwargs={}):  # use beam search
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', -1)
    split = eval_kwargs.get('eval_split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 3)
    batch_size = eval_kwargs.get('batch_size', 1)
    print_beam_candidate = eval_kwargs.get('print_beam_candidate', 1)
    seq_length = eval_kwargs.get('seq_length', 16)
    caption_model = eval_kwargs.get('caption_model', 'fc')
    use_cuda = eval_kwargs.get('use_cuda', 0)
    rank = eval_kwargs.get('rank', 0)
    feature_type = eval_kwargs.get('feature_type', 'resnet')
    feat_mask = eval_kwargs.get('feat_mask', 0)
    num_feat_array = sum(feat_mask)

    print('batch_size: ' + str(batch_size))
    print('beam_size: ' + str(beam_size))
    print('caption_model: ' + caption_model)
    print('beam search to generate caption for : ' + split)
    print('use_cuda: ' + str(use_cuda))

    assert beam_size > 1, 'beam_size not correct'

    # Make sure in the evaluation mode
    for model in model_list:
        model.eval()
    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    num_model = len(model_list)

    while True:
        # fetch a batch of data
        data = loader.get_batch(split, batch_size)
        n = n + batch_size

        if feature_type == 'feat_array':
            fc_feat_array_temp = data['fc_feats_array']
            att_feat_array_temp = data['att_feats_array']
            # num_feat_array = len(fc_feat_array_temp)

            fc_feat_array = [[] for _ in range(num_feat_array)]
            att_feat_array = [[] for _ in range(num_feat_array)]
            for feat_id in range(num_feat_array):
                fc_temp = fc_feat_array_temp[feat_id][np.arange(loader.batch_size) * loader.seq_per_img]
                att_temp = att_feat_array_temp[feat_id][np.arange(loader.batch_size) * loader.seq_per_img]
                if use_cuda:
                    fc_feat_array[feat_id] = Variable(torch.from_numpy(fc_temp), volatile=True).cuda()
                    att_feat_array[feat_id] = Variable(torch.from_numpy(att_temp), volatile=True).cuda()
                else:
                    fc_feat_array[feat_id] = Variable(torch.from_numpy(fc_temp), volatile=True)
                    att_feat_array[feat_id] = Variable(torch.from_numpy(att_temp), volatile=True)

            tmp = [data['obj_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['mil_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['matching_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['top_words'][np.arange(loader.batch_size) * loader.seq_per_img]
                   ]
            if use_cuda:
                tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            else:
                tmp = [Variable(torch.from_numpy(_), volatile=True) for _ in tmp]
            obj_feats, mil_feats, matching_feats, labels, masks, top_words = tmp
        else:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['obj_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['mil_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['matching_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
            if use_cuda:
                tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            else:
                tmp = [Variable(torch.from_numpy(_), volatile=True) for _ in tmp]
            fc_feats, att_feats, obj_feats, mil_feats, matching_feats = tmp
            fc_feat_size = fc_feats.size(1)

        seq = torch.LongTensor(seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(seq_length, batch_size)
        # lets process every image independently for now, for simplicity
        if use_cuda:
            seq = seq.cuda()
            seqLogprobs = seqLogprobs.cuda()

        top_seq = []
        top_prob = [[] for _ in range(batch_size)]

        done_beams = [[] for _ in range(batch_size)]

        for k in range(batch_size):
            if feature_type == 'feat_array':
                fc_feat_size_list = []
                for fc in fc_feat_array:
                    fc_feat_size_list.append(fc.size(1))

                fc_feats_current = []
                att_feats_current = []
                for feat_id in range(num_feat_array):
                    fc_feats_current.append(
                        fc_feat_array[feat_id][k].unsqueeze(0).expand(beam_size, fc_feat_size_list[feat_id]))
                    att_feats_current.append(
                        att_feat_array[feat_id][k].unsqueeze(0).expand(beam_size, att_feat_array[feat_id].size(1),
                                                                       att_feat_array[feat_id].size(2)))

                for feat_id in range(num_feat_array):
                    fc_feats_current[feat_id] = fc_feats_current[feat_id].contiguous()
                    att_feats_current[feat_id] = att_feats_current[feat_id].contiguous()

            else:
                fc_feats_current = fc_feats[k].unsqueeze(0).expand(beam_size, fc_feat_size)
                att_feats_current = att_feats[k].unsqueeze(0).expand(beam_size, att_feats.size(1), att_feats.size(2))
                fc_feats_current = fc_feats_current.contiguous()
                att_feats_current = att_feats_current.contiguous()

            mil_feats_current = mil_feats[k].unsqueeze(0).expand(beam_size, mil_feats.size(1))
            matching_feats_current = matching_feats[k].unsqueeze(0).expand(beam_size, matching_feats.size(1))

            mil_feats_current = mil_feats_current.contiguous()
            matching_feats_current = matching_feats_current.contiguous()

            state_list = []
            for model in model_list:
                if feature_type == 'feat_array':
                    state_list.append(model.get_init_state(fc_feats_current))
                else:
                    state_list.append(model.get_init_state(fc_feats_current))

            thought_vector_list = []
            gv_reason_list = []
            gv_decoder_list = []

            if caption_model == 'show_tell':
                pass

            elif caption_model == 'review_net':
                for i in range(num_model):
                    thought_vectors, _, state_new = model_list[i].get_thought_vectors(fc_feats_current,
                                                                                      att_feats_current,
                                                                                      mil_feats_current,
                                                                                      matching_feats_current,
                                                                                      state_list[i])
                    thought_vector_list.append(thought_vectors)
                    state_list[i] = state_new

            elif caption_model == 'recurrent_fusion_model':
                for i in range(num_model):
                    thought_vectors, _, state_new = model_list[i].get_thought_vectors(
                        fc_feats_current, att_feats_current, mil_feats_current, matching_feats_current, state_list[i]
                    )
                    thought_vector_list.append(thought_vectors)
                    state_list[i] = state_new
            else:
                raise Exception("caption_model not supported: {}".format(caption_model))

            beam_seq = torch.LongTensor(seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam

            for t in range(seq_length + 1):
                if t == 0:  # input <bos>
                    if feature_type == 'feat_array':
                        it = fc_feats_current[0].data.new(beam_size).long().zero_()
                    else:
                        it = fc_feats.data.new(beam_size).long().zero_()
                    xt_list = get_xt_list(model_list, it)
                    # xt = self.embed(Variable(it, requires_grad=False))
                    # xt = self.img_embed(fc_feats[k:k+1]).expand(beam_size, self.input_encoding_size)
                else:
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
                    # new_state = [_.clone() for _ in state]
                    new_state_list = [None] * num_model
                    for model_index in range(num_model):
                        state = state_list[model_index]
                        new_state_list[model_index] = [_.clone() for _ in state]

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
                        for model_index in range(num_model):
                            new_state = new_state_list[model_index]
                            state = state_list[model_index]
                            for state_ix in range(len(new_state)):
                                # copy over state in previous beam q to new beam at vix
                                new_state[state_ix][0, vix] = state[state_ix][0, v['q']]  # dimension one is time step

                            new_state_list[model_index] = new_state

                        # append new end terminal at the end of this beam
                        beam_seq[t - 1, vix] = v['c']  # c'th word is the continuation
                        beam_seq_logprobs[t - 1, vix] = v['r']  # the raw logprob here
                        beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam

                        if v['c'] == 0 or t == seq_length:
                            done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                  'logps': beam_seq_logprobs[:, vix].clone(),
                                                  'p': beam_logprobs_sum[vix]
                                                  })

                    # encode as vectors
                    it = beam_seq[t - 1]
                    # xt = self.embed(Variable(it.cuda()))
                    if use_cuda:
                        xt_list = get_xt_list(model_list, it.cuda())
                    else:
                        xt_list = get_xt_list(model_list, it)

                if t >= 1:
                    state_list = new_state_list
                if feature_type == 'feat_array':
                    logit_list, state_list, logprobs = model_ensemble_feat_array_one_step(model_list, xt_list,
                                                                                          fc_feats_current,
                                                                                          att_feats_current, mil_feats,
                                                                                          matching_feats, state_list,
                                                                                          thought_vector_list,
                                                                                          caption_model)
                else:
                    logit_list, state_list, logprobs = model_ensemble_one_step(
                        model_list, xt_list, fc_feats_current, att_feats_current, mil_feats_current,
                        matching_feats_current, state_list, thought_vector_list, gv_decoder_list, caption_model)

            done_beams[k] = sorted(done_beams[k], key=lambda x: -x['p'])
            seq[:, k] = done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = done_beams[k][0]['logps']

            # save result
            l = len(done_beams[k])
            top_seq_cur = torch.LongTensor(l, seq_length).zero_()

            for temp_index in range(l):
                top_seq_cur[temp_index] = done_beams[k][temp_index]['seq'].clone()
                top_prob[k].append(done_beams[k][temp_index]['p'])

            top_seq.append(top_seq_cur)

        seq = seq.transpose(0, 1)
        seqLogprobs = seqLogprobs.transpose(0, 1)

        # top_seq, top_prob

        # seq, _, top_seq, top_prob = model.sample(fc_feats, att_feats, mil_feats, matching_feats, eval_kwargs)
        if print_beam_candidate >= 1:
            for batch_index in range(batch_size):
                image_id = data['infos'][batch_index]['id']
                sents = utils.decode_sequence(loader.get_vocab(), top_seq[batch_index])
                sents_to_print = {}
                for index_temp in range(len(sents)):
                    cur_sent = sents[index_temp]
                    if cur_sent in sents_to_print:
                        if top_prob[batch_index][index_temp] > sents_to_print[cur_sent]:
                            sents_to_print[cur_sent] = top_prob[batch_index][index_temp]
                    else:
                        sents_to_print[cur_sent] = top_prob[batch_index][index_temp]

                sorted_sents_to_print = sorted(sents_to_print.items(), key=operator.itemgetter(1))
                sorted_sents_to_print.reverse()

                for beam_index in range(beam_size):
                    print('%d\t%s\t%s' % (
                    image_id, sorted_sents_to_print[beam_index][1], sorted_sents_to_print[beam_index][0]))

        # set_trace()
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        seqLogprobs = torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)
        if use_cuda:
            log_probs = torch.sum(seqLogprobs * (seq > 0).type(torch.cuda.FloatTensor), 1)
        else:
            log_probs = torch.sum(seqLogprobs * (seq > 0).type(torch.FloatTensor), 1)
        for k, sent in enumerate(sents):
            # entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'log_prob': log_probs[k][0]}
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'log_prob': log_probs[k]}
            predictions.append(entry)
            if print_beam_candidate < 1:
                print('%s\t%s\t%s' % (entry['image_id'], entry['log_prob'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if data['bounds']['wrapped']:
            break
        if n >= num_images >= 0:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, 'ensemble_' + eval_kwargs['caption_model'], split)

    # Switch back to training mode
    # model.train()
    return loss_sum / loss_evals, predictions, lang_stats


def move_list_to_gpu(source_list, gpu_id):
    target_list = []
    for e in source_list:
        target_list.append(e.cuda(gpu_id))
    return target_list


def eval_ensemble_greedy(model_list, loader, eval_kwargs={}, model_id_2_gpu_id={},
                         gpu_id_2_model_id={}):  # use greedy search
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', -1)
    split = eval_kwargs.get('eval_split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 3)
    batch_size = eval_kwargs.get('batch_size', 1)
    print_beam_candidate = eval_kwargs.get('print_beam_candidate', 1)
    seq_length = eval_kwargs.get('seq_length', 16)
    caption_model = eval_kwargs.get('caption_model', 'fc')
    use_cuda = eval_kwargs.get('use_cuda', 0)
    rank = eval_kwargs.get('use_cuda', 0)
    feature_type = eval_kwargs.get('feature_type', 'resnet')
    eval_ensemble_multi_gpu = eval_kwargs.get('eval_ensemble_multi_gpu', 0)

    print('batch_size: ' + str(batch_size))
    print('beam_size: ' + str(beam_size))
    print('caption_model: ' + caption_model)
    print('greedy search to generate caption for : ' + split)
    print('use_cuda: ' + str(use_cuda))

    # assert beam_size = 1, 'beam_size not correct'

    # Make sure in the evaluation mode
    for model in model_list:
        model.eval()
    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    num_model = len(model_list)

    while True:
        # fetch a batch of data
        data = loader.get_batch(split, batch_size)
        n = n + batch_size

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        if feature_type == 'feat_array':
            fc_feat_array_temp = data['fc_feats_array']
            att_feat_array_temp = data['att_feats_array']
            num_feat_array = len(fc_feat_array_temp)

            fc_feat_array = [[] for _ in range(num_feat_array)]
            att_feat_array = [[] for _ in range(num_feat_array)]
            for feat_id in range(num_feat_array):
                fc_temp = fc_feat_array_temp[feat_id][np.arange(loader.batch_size) * loader.seq_per_img]
                att_temp = att_feat_array_temp[feat_id][np.arange(loader.batch_size) * loader.seq_per_img]
                if use_cuda:
                    fc_feat_array[feat_id] = Variable(torch.from_numpy(fc_temp), volatile=True).cuda()
                    att_feat_array[feat_id] = Variable(torch.from_numpy(att_temp), volatile=True).cuda()
                else:
                    fc_feat_array[feat_id] = Variable(torch.from_numpy(fc_temp), volatile=True)
                    att_feat_array[feat_id] = Variable(torch.from_numpy(att_temp), volatile=True)

            tmp = [data['obj_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['mil_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['matching_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['masks'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['top_words'][np.arange(loader.batch_size) * loader.seq_per_img]
                   ]
            if use_cuda:
                tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            else:
                tmp = [Variable(torch.from_numpy(_), volatile=True) for _ in tmp]
            obj_feats, mil_feats, matching_feats, labels, masks, top_words = tmp

            fc_feat_array_multi_gpu = {}
            att_feat_array_multi_gpu = {}
            obj_feats_multi_gpu = {}
            mil_feats_multi_gpu = {}
            matching_feats_multi_gpu = {}
            labels_multi_gpu = {}
            masks_multi_gpu = {}
            top_words_multi_gpu = {}
            if eval_ensemble_multi_gpu:
                for gpu_id in gpu_id_2_model_id.keys():
                    fc_feat_array_multi_gpu[gpu_id] = move_list_to_gpu(fc_feat_array, gpu_id)
                    att_feat_array_multi_gpu[gpu_id] = move_list_to_gpu(att_feat_array, gpu_id)
                    obj_feats_multi_gpu[gpu_id] = obj_feats.cuda(gpu_id)
                    mil_feats_multi_gpu[gpu_id] = mil_feats.cuda(gpu_id)
                    matching_feats_multi_gpu[gpu_id] = matching_feats.cuda(gpu_id)
                    labels_multi_gpu[gpu_id] = labels.cuda(gpu_id)
                    masks_multi_gpu[gpu_id] = masks.cuda(gpu_id)
                    top_words_multi_gpu[gpu_id] = top_words.cuda(gpu_id)

        else:
            tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['obj_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['mil_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
                   data['matching_feats'][np.arange(loader.batch_size) * loader.seq_per_img]]
            if use_cuda:
                tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
            else:
                tmp = [Variable(torch.from_numpy(_), volatile=True) for _ in tmp]
            fc_feats, att_feats, obj_feats, mil_feats, matching_feats = tmp
        # forward the model to also get generated samples for each image

        seq = []
        seqLogprobs = []

        state_list = []
        if feature_type == 'feat_array':
            if eval_ensemble_multi_gpu:
                for model_id, model in enumerate(model_list):
                    gpu_id = model_id_2_gpu_id[model_id]
                    state_list.append(model.get_init_state(fc_feat_array_multi_gpu[gpu_id]))
            else:
                for model in model_list:
                    state_list.append(model.get_init_state(fc_feat_array))
        else:
            for model in model_list:
                state_list.append(model.get_init_state(fc_feats))

        thought_vector_list = []
        gv_reason_list = []
        gv_decoder_list = []
        if caption_model == 'show_tell':
            pass

        elif caption_model == 'review_net':
            for i in range(num_model):
                thought_vectors, _, state_new = model_list[i].get_thought_vectors(fc_feats, att_feats, mil_feats,
                                                                                  matching_feats, state_list[i])
                thought_vector_list.append(thought_vectors)
                state_list[i] = state_new

        elif caption_model == 'recurrent_fusion_model':
            state_list_temp = []
            # print('len state_list' + str(len(state_list)))
            if eval_ensemble_multi_gpu:
                for i in range(num_model):
                    gpu_id = model_id_2_gpu_id[i]
                    with torch.cuda.device(gpu_id):
                        thought_vectors, _, state_list_new = model_list[i].get_thought_vectors(
                            fc_feat_array_multi_gpu[gpu_id], att_feat_array_multi_gpu[gpu_id],
                            mil_feats_multi_gpu[gpu_id], matching_feats_multi_gpu[gpu_id], state_list[i]
                        )
                        thought_vector_list.append(thought_vectors)  # list of list
                        state_list_temp.append(state_list_new)

                state_list = state_list_temp
            else:
                for i in range(num_model):
                    thought_vectors, _, state_list_new = model_list[i].get_thought_vectors(
                        fc_feat_array, att_feat_array, mil_feats, matching_feats, state_list[i]
                    )
                    thought_vector_list.append(thought_vectors)  # list of list
                    state_list_temp.append(state_list_new)

                state_list = state_list_temp

        else:
            raise Exception("caption_model not supported: {}".format(caption_model))

        for t in range(seq_length + 1):
            if t == 0:  # input <bos>
                if feature_type == 'feat_array':
                    it = fc_feat_array[0].data.new(batch_size).long().zero_()
                else:
                    it = fc_feats.data.new(batch_size).long().zero_()
            else:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
                # top_log_probs, _ = torch.topk(logprobs.data, 10, 1, True, True)
                # print(torch.exp(top_log_probs))

            if eval_ensemble_multi_gpu:
                xt_list = get_xt_list_multi_gpu(model_list, it, model_id_2_gpu_id)
            else:
                xt_list = get_xt_list(model_list, it)

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

            if feature_type == 'feat_array':
                if eval_ensemble_multi_gpu:
                    logit_list, state_list, logprobs = model_ensemble_feat_array_one_step_multi_gpu(model_list, xt_list,
                                                                                                    fc_feat_array_multi_gpu,
                                                                                                    att_feat_array_multi_gpu,
                                                                                                    mil_feats_multi_gpu,
                                                                                                    matching_feats_multi_gpu,
                                                                                                    state_list,
                                                                                                    thought_vector_list,
                                                                                                    model_id_2_gpu_id)
                else:

                    logit_list, state_list, logprobs = model_ensemble_feat_array_one_step(model_list, xt_list,
                                                                                          fc_feat_array,
                                                                                          att_feat_array, mil_feats,
                                                                                          matching_feats, state_list,
                                                                                          thought_vector_list,
                                                                                          caption_model
                                                                                          )
            else:
                logit_list, state_list, logprobs = model_ensemble_one_step(model_list, xt_list, fc_feats,
                                                                           att_feats, mil_feats,
                                                                           matching_feats, state_list,
                                                                           thought_vector_list, gv_decoder_list,
                                                                           caption_model)
        seq = torch.cat([_.unsqueeze(1) for _ in seq], 1)
        seqLogprobs = torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)
        log_probs = torch.sum(seqLogprobs * (seq > 0).type(torch.cuda.FloatTensor), 1)
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        assert len(sents) == batch_size
        for k, sent in enumerate(sents):
            # entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'log_prob': log_probs[k][0]}
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'log_prob': log_probs[k]}
            predictions.append(entry)
            print('%s\t%f\t%s' % (entry['image_id'], entry['log_prob'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if data['bounds']['wrapped']:
            break
        if n >= num_images >= 0:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, 'ensemble_' + eval_kwargs['caption_model'], split)

    # Switch back to training mode
    # model.train()
    return loss_sum / loss_evals, predictions, lang_stats


# one step, review net with different feature types
def model_ensemble_one_step_diff_feat(model_all, xt_all, fc_feat_array, att_feat_array, mil_feats, matching_feats,
                                      state_all,
                                      thought_vector_all):
    logit_all = []
    state_all_new = []
    num_model = 0
    for feat_id, model_list in enumerate(model_all):
        logit_list = []
        state_list_new = []
        for i, model in enumerate(model_list):
            xt = xt_all[feat_id][i]
            fc_feats = fc_feat_array[feat_id]
            thought_vector = thought_vector_all[feat_id][i]
            state = state_all[feat_id][i]
            logit, state_new = model.one_time_step(xt, fc_feats, thought_vector, mil_feats, matching_feats, state)

            logit_list.append(logit)
            state_list_new.append(state_new)
            num_model = num_model + 1

        logit_all.append(logit_list)
        state_all_new.append(state_list_new)

    # logit_mean = sum([sum(e) for e in logit_all]) / num_model
    logit_mean = logit_all[0][0].clone()
    logit_mean.data.zero_()
    for logit_list in logit_all:
        for l in logit_list:
            logit_mean = logit_mean + l

    logit_mean = logit_mean / num_model
    logprob = F.log_softmax(logit_mean)

    return logit_all, state_all_new, logprob


def get_xt_all_diff_feat(model_all, it):
    xt_all = []
    for model_list in model_all:
        xt_list = []
        for model in model_list:
            xt = model.embed(Variable(it, requires_grad=False))
            xt_list.append(xt)
        xt_all.append(xt_list)
    return xt_all


def eval_ensemble_diff_feat_greedy(model_all, loader, eval_kwargs={}):  # use greedy search
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', -1)
    split = eval_kwargs.get('eval_split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 3)
    batch_size = eval_kwargs.get('batch_size', 1)
    print_beam_candidate = eval_kwargs.get('print_beam_candidate', 1)
    seq_length = eval_kwargs.get('seq_length', 16)
    caption_model = eval_kwargs.get('caption_model', 'fc')
    use_cuda = eval_kwargs.get('use_cuda', 0)
    rank = eval_kwargs.get('rank', 0)
    feature_type = eval_kwargs.get('feature_type', 'resnet')  # feat_array

    print('---------- ensemble_diff_feat_greed: ' + split)

    assert (feature_type == 'feat_array')
    assert (caption_model == 'review_net')

    print('batch_size: ' + str(batch_size))
    print('beam_size: ' + str(beam_size))
    print('caption_model: ' + caption_model)
    print('greedy search to generate caption for : ' + split)
    print('use_cuda: ' + str(use_cuda))

    # assert beam_size = 1, 'beam_size not correct'

    num_model = 0
    # Make sure in the evaluation mode
    for model_list in model_all:
        for model in model_list:
            model.eval()
            num_model = num_model + 1
    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []

    while True:
        # fetch a batch of data
        data = loader.get_batch(split, batch_size)
        n = n + batch_size

        fc_feat_array_temp = data['fc_feats_array']
        att_feat_array_temp = data['att_feats_array']
        num_feat_array = len(fc_feat_array_temp)
        assert (len(model_all) == num_feat_array)

        fc_feat_array = [[] for _ in range(num_feat_array)]
        att_feat_array = [[] for _ in range(num_feat_array)]
        for feat_id in range(num_feat_array):
            fc_temp = fc_feat_array_temp[feat_id][np.arange(loader.batch_size) * loader.seq_per_img]
            att_temp = att_feat_array_temp[feat_id][np.arange(loader.batch_size) * loader.seq_per_img]
            if use_cuda:
                fc_feat_array[feat_id] = Variable(torch.from_numpy(fc_temp), volatile=True).cuda()
                att_feat_array[feat_id] = Variable(torch.from_numpy(att_temp), volatile=True).cuda()
            else:
                fc_feat_array[feat_id] = Variable(torch.from_numpy(fc_temp), volatile=True)
                att_feat_array[feat_id] = Variable(torch.from_numpy(att_temp), volatile=True)

        tmp = [data['obj_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['mil_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['matching_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['masks'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['top_words'][np.arange(loader.batch_size) * loader.seq_per_img]
               ]
        obj_feats, mil_feats, matching_feats, labels, masks, top_words = tmp

        seq = []
        seqLogprobs = []

        state_all = []

        for feat_id, model_list in enumerate(model_all):
            state_list = []
            for model in model_list:
                state_list.append(model.get_init_state(fc_feat_array[feat_id]))
            state_all.append(state_list)

        thought_vector_all = []
        for feat_id, model_list in enumerate(model_all):
            thought_vector_list = []
            for i, model in enumerate(model_list):
                thought_vectors, _, state_new = model.get_thought_vectors(fc_feat_array[feat_id],
                                                                          att_feat_array[feat_id],
                                                                          mil_feats,
                                                                          matching_feats,
                                                                          state_all[feat_id][i])
                thought_vector_list.append(thought_vectors)
                state_all[feat_id][i] = state_new

            thought_vector_all.append(thought_vector_list)

        for t in range(seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feat_array[0][0].data.new(batch_size).long().zero_()
            else:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
                # top_log_probs, _ = torch.topk(logprobs.data, 10, 1, True, True)
                # print(torch.exp(top_log_probs))

            xt_all = get_xt_all_diff_feat(model_all, it)

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

            logit_list, state_all, logprobs = model_ensemble_one_step_diff_feat(model_all, xt_all, fc_feat_array,
                                                                                att_feat_array, mil_feats,
                                                                                matching_feats, state_all,
                                                                                thought_vector_all)
        seq = torch.cat([_.unsqueeze(1) for _ in seq], 1)
        seqLogprobs = torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)
        log_probs = torch.sum(seqLogprobs * (seq > 0).type(torch.cuda.FloatTensor), 1)
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        assert len(sents) == batch_size
        for k, sent in enumerate(sents):
            # entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'log_prob': log_probs[k][0]}
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'log_prob': log_probs[k]}
            predictions.append(entry)
            print('%s\t%f\t%s' % (entry['image_id'], entry['log_prob'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if data['bounds']['wrapped']:
            break
        if n >= num_images >= 0:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, 'ensemble_' + eval_kwargs['caption_model'], split)

    # Switch back to training mode
    # model.train()
    return loss_sum / loss_evals, predictions, lang_stats


def eval_ensemble_diff_feat_beam_search(model_all, loader, eval_kwargs={}):  # use beam search
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', -1)
    split = eval_kwargs.get('eval_split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 3)
    batch_size = eval_kwargs.get('batch_size', 1)
    print_beam_candidate = eval_kwargs.get('print_beam_candidate', 1)
    seq_length = eval_kwargs.get('seq_length', 16)
    caption_model = eval_kwargs.get('caption_model', 'fc')
    use_cuda = eval_kwargs.get('use_cuda', 0)
    rank = eval_kwargs.get('rank', 0)
    feature_type = eval_kwargs.get('feature_type', 'resnet')
    feat_mask = eval_kwargs.get('feat_mask', 0)
    num_feat_array = sum(feat_mask)

    print('batch_size: ' + str(batch_size))
    print('beam_size: ' + str(beam_size))
    print('caption_model: ' + caption_model)
    print('beam search to generate caption for : ' + split)
    print('use_cuda: ' + str(use_cuda))

    assert (feature_type == 'feat_array')
    assert beam_size > 1, 'beam_size not correct'

    # Make sure in the evaluation mode
    num_model = 0
    for model_list in model_all:
        for model in model_list:
            model.eval()
            num_model = num_model + 1
    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []

    while True:
        # fetch a batch of data
        data = loader.get_batch(split, batch_size)
        n = n + batch_size
        fc_feat_array_temp = data['fc_feats_array']
        att_feat_array_temp = data['att_feats_array']
        # num_feat_array = len(fc_feat_array_temp)

        fc_feat_array = [[] for _ in range(num_feat_array)]
        att_feat_array = [[] for _ in range(num_feat_array)]
        for feat_id in range(num_feat_array):
            fc_temp = fc_feat_array_temp[feat_id][np.arange(loader.batch_size) * loader.seq_per_img]
            att_temp = att_feat_array_temp[feat_id][np.arange(loader.batch_size) * loader.seq_per_img]
            if use_cuda:
                fc_feat_array[feat_id] = Variable(torch.from_numpy(fc_temp), volatile=True).cuda()
                att_feat_array[feat_id] = Variable(torch.from_numpy(att_temp), volatile=True).cuda()
            else:
                fc_feat_array[feat_id] = Variable(torch.from_numpy(fc_temp), volatile=True)
                att_feat_array[feat_id] = Variable(torch.from_numpy(att_temp), volatile=True)

        tmp = [data['obj_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['mil_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['matching_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['labels'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['masks'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['top_words'][np.arange(loader.batch_size) * loader.seq_per_img]
               ]
        if use_cuda:
            tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        else:
            tmp = [Variable(torch.from_numpy(_), volatile=True) for _ in tmp]
        obj_feats, mil_feats, matching_feats, labels, masks, top_words = tmp

        seq = torch.LongTensor(seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(seq_length, batch_size)
        # lets process every image independently for now, for simplicity
        if use_cuda:
            seq = seq.cuda()
            seqLogprobs = seqLogprobs.cuda()

        top_seq = []
        top_prob = [[] for _ in range(batch_size)]

        done_beams = [[] for _ in range(batch_size)]

        for k in range(batch_size):
            fc_feat_size_list = []
            for fc in fc_feat_array:
                fc_feat_size_list.append(fc.size(1))

            fc_feats_current = []
            att_feats_current = []
            for feat_id in range(num_feat_array):
                fc_feats_current.append(
                    fc_feat_array[feat_id][k].unsqueeze(0).expand(beam_size, fc_feat_size_list[feat_id]))
                att_feats_current.append(
                    att_feat_array[feat_id][k].unsqueeze(0).expand(beam_size, att_feat_array[feat_id].size(1),
                                                                   att_feat_array[feat_id].size(2)))

            for feat_id in range(num_feat_array):
                fc_feats_current[feat_id] = fc_feats_current[feat_id].contiguous()
                att_feats_current[feat_id] = att_feats_current[feat_id].contiguous()

            mil_feats_current = mil_feats[k].unsqueeze(0).expand(beam_size, mil_feats.size(1))
            matching_feats_current = matching_feats[k].unsqueeze(0).expand(beam_size, matching_feats.size(1))

            mil_feats_current = mil_feats_current.contiguous()
            matching_feats_current = matching_feats_current.contiguous()

            state_all = []
            for feat_id, model_list in enumerate(model_all):
                state_list = []
                for model in model_list:
                    state_list.append(model.get_init_state(fc_feats_current[feat_id]))
                state_all.append(state_list)

            thought_vector_all = []
            for feat_id, model_list in enumerate(model_all):
                thought_vector_list = []
                for i, model in enumerate(model_list):
                    thought_vectors, _, state_new = model.get_thought_vectors(fc_feats_current[feat_id],
                                                                              att_feats_current[feat_id],
                                                                              mil_feats_current,
                                                                              matching_feats_current,
                                                                              state_all[feat_id][i])
                    thought_vector_list.append(thought_vectors)
                    state_all[feat_id][i] = state_new

                thought_vector_all.append(thought_vector_list)

            beam_seq = torch.LongTensor(seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam

            for t in range(seq_length + 1):
                if t == 0:  # input <bos>
                    it = fc_feats_current[0].data.new(beam_size).long().zero_()
                    xt_all = get_xt_all_diff_feat(model_all, it)
                    # xt = self.embed(Variable(it, requires_grad=False))
                    # xt = self.img_embed(fc_feats[k:k+1]).expand(beam_size, self.input_encoding_size)
                else:
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
                    # new_state = [_.clone() for _ in state]
                    new_state_all = []
                    for feat_id in range(num_feat_array):
                        state_list = state_all[feat_id]
                        state_list_new = []
                        for state in state_list:  # state[0] size: 1 * 3 * rnn_size
                            state_temp = [_.clone() for _ in state]
                            state_list_new.append(state_temp)

                        new_state_all.append(state_list_new)

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
                        # for model_index in range(num_model):
                        #     new_state = new_state_list[model_index]
                        #     state = state_list[model_index]
                        #     for state_ix in range(len(new_state)):
                        #         # copy over state in previous beam q to new beam at vix
                        #         new_state[state_ix][0, vix] = state[state_ix][0, v['q']]  # dimension one is time step
                        #
                        #     new_state_list[model_index] = new_state

                        for feat_id in range(num_feat_array):
                            new_state_list = new_state_all[feat_id]
                            state_list = state_all[feat_id]
                            for model_index in range(len(state_list)):
                                state = state_list[model_index]
                                new_state = new_state_list[model_index]
                                for state_ix in range(len(new_state)):
                                    new_state[state_ix][0, vix] = state[state_ix][0, v['q']]

                        # append new end terminal at the end of this beam
                        beam_seq[t - 1, vix] = v['c']  # c'th word is the continuation
                        beam_seq_logprobs[t - 1, vix] = v['r']  # the raw logprob here
                        beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam

                        if v['c'] == 0 or t == seq_length:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                  'logps': beam_seq_logprobs[:, vix].clone(),
                                                  'p': beam_logprobs_sum[vix]
                                                  })

                    # encode as vectors
                    it = beam_seq[t - 1]
                    # xt = self.embed(Variable(it.cuda()))
                    if use_cuda:
                        xt_all = get_xt_all_diff_feat(model_all, it.cuda())
                    else:
                        xt_all = get_xt_all_diff_feat(model_all, it)

                if t >= 1:
                    state_all = new_state_all
                logit_list, state_all, logprobs = model_ensemble_one_step_diff_feat(model_all, xt_all, fc_feats_current,
                                                                                    att_feats_current,
                                                                                    mil_feats_current,
                                                                                    matching_feats_current, state_all,
                                                                                    thought_vector_all)
                # output, state = self.core(xt, att_feats_current, mil_feats_current, matching_feats_current, state)
                # logprobs = F.log_softmax(self.logit(output))

            done_beams[k] = sorted(done_beams[k], key=lambda x: -x['p'])
            seq[:, k] = done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = done_beams[k][0]['logps']

            # save result
            l = len(done_beams[k])
            top_seq_cur = torch.LongTensor(l, seq_length).zero_()

            for temp_index in range(l):
                top_seq_cur[temp_index] = done_beams[k][temp_index]['seq'].clone()
                top_prob[k].append(done_beams[k][temp_index]['p'])

            top_seq.append(top_seq_cur)

        seq = seq.transpose(0, 1)
        seqLogprobs = seqLogprobs.transpose(0, 1)

        # top_seq, top_prob

        # seq, _, top_seq, top_prob = model.sample(fc_feats, att_feats, mil_feats, matching_feats, eval_kwargs)
        if print_beam_candidate >= 1:
            for batch_index in range(batch_size):
                image_id = data['infos'][batch_index]['id']
                sents = utils.decode_sequence(loader.get_vocab(), top_seq[batch_index])
                sents_to_print = {}
                for index_temp in range(len(sents)):
                    cur_sent = sents[index_temp]
                    if cur_sent in sents_to_print:
                        if top_prob[batch_index][index_temp] > sents_to_print[cur_sent]:
                            sents_to_print[cur_sent] = top_prob[batch_index][index_temp]
                    else:
                        sents_to_print[cur_sent] = top_prob[batch_index][index_temp]

                sorted_sents_to_print = sorted(sents_to_print.items(), key=operator.itemgetter(1))
                sorted_sents_to_print.reverse()

                for beam_index in range(beam_size):
                    print('%d\t%s\t%s' % (
                    image_id, sorted_sents_to_print[beam_index][1], sorted_sents_to_print[beam_index][0]))

        # set_trace()
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        seqLogprobs = torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)
        if use_cuda:
            log_probs = torch.sum(seqLogprobs * (seq > 0).type(torch.cuda.FloatTensor), 1)
        else:
            log_probs = torch.sum(seqLogprobs * (seq > 0).type(torch.FloatTensor), 1)
        for k, sent in enumerate(sents):
            # entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'log_prob': log_probs[k][0]}
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'log_prob': log_probs[k]}
            predictions.append(entry)
            if print_beam_candidate < 1:
                print('%s\t%s\t%s' % (entry['image_id'], entry['log_prob'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if data['bounds']['wrapped']:
            break
        if n >= num_images >= 0:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, 'ensemble_' + eval_kwargs['caption_model'], split)

    # Switch back to training mode
    # model.train()
    return loss_sum / loss_evals, predictions, lang_stats
