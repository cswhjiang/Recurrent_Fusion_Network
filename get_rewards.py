# -*- coding: utf-8 -*-

import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch
from torch.autograd import Variable

import sys

sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD
from pyciderevalcap.bleuD.bleuD import BleuD
from pyciderevalcap.spiceD.spiceD import SpiceD

CiderD_scorer = CiderD(df='coco-train-idxs')


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


def array_to_seq(arr, end_word, idx_to_word):
    out = ''
    for i in range(len(arr)):
        if arr[i] != 0:
            out += idx_to_word[str(arr[i])] + ' '
            if arr[i] == end_word:
                break
    return out.strip()


def compute_reward(idx_to_word, gen_result, greedy_res, data, opt):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])
    gen_result = gen_result.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()

    res = OrderedDict()

    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_spice = OrderedDict()
    for i in range(batch_size):
        res_spice[i] = [array_to_seq(gen_result[i], 0, idx_to_word)]
    for i in range(batch_size):
        res_spice[batch_size + i] = [array_to_seq(greedy_res[i], 0, idx_to_word)]

    gts_spice = OrderedDict()
    gts_data = data['gts']
    for i in range(len(gts_data)):
        gts_spice[i] = [array_to_seq(gts_data[i][j], 0, idx_to_word) for j in range(len(gts_data[i]))]

    res_spice = {i: res_spice[i] for i in range(2 * batch_size)}
    gts_spice = {i: gts_spice[i % batch_size // seq_per_img] for i in range(2 * batch_size)}

    res = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}

    cider_mean, cider_scores = CiderD_scorer.compute_score(gts, res)

    if opt.bleu4_weight > 0:
        start_bleu = time.time()
        bleu_score_mean, bleu_scores = BleuD(4).compute_score(gts, res)
        bleu4_scores = np.array(bleu_scores[3])
        end_bleu = time.time()
        print("bleu time: {:.3f}".format((end_bleu - start_bleu)))
    else:
        bleu_score_mean = np.zeros(4)
        bleu4_scores = np.zeros_like(cider_scores)

    if opt.spice_weight > 0:
        start_spice = time.time()
        SpiceD_scorer = SpiceD()
        spice_score_mean, spice_scores = SpiceD_scorer.compute_score(gts_spice, res_spice, opt.ip, opt.port)
        spice_scores = np.asarray(spice_scores)
        end_spice = time.time()
        print("spice time: {:.3f}".format((end_spice - start_spice)))
    else:
        spice_score_mean = 0
        spice_scores = np.zeros_like(cider_scores)

    print('Cider: {:.3f}, Spice: {:.3f}, bleu_1: {:.3f}, bleu_2: {:.3f}, bleu_3: {:.3f}, bleu_4: {:.3f}'.format(
        cider_mean, spice_score_mean, bleu_score_mean[0], bleu_score_mean[1], bleu_score_mean[2], bleu_score_mean[3]))

    if opt.use_baseline:
        cider_scores = cider_scores[:batch_size] - cider_scores[batch_size:]
        bleu4_scores = bleu4_scores[:batch_size] - bleu4_scores[batch_size:]
        spice_scores = spice_scores[:batch_size] - spice_scores[batch_size:]

    else:
        cider_scores = cider_scores[:batch_size]
        bleu4_scores = bleu4_scores[:batch_size]
        spice_scores = spice_scores[:batch_size]

    scores_combine = bleu4_scores * opt.bleu4_weight + cider_scores * opt.cider_weight + spice_scores * opt.spice_weight
    rewards = np.repeat(scores_combine[:, np.newaxis], gen_result.shape[1], 1)

    return rewards


def get_self_critical_reward_feat_array(idx_to_word, model, fc_feat_array, att_feat_array,
                                        data, gen_result, opt):
    # get greedy decoding baseline

    fc_feat_array_var = [Variable(fc_feat.data, volatile=True) for fc_feat in fc_feat_array]
    att_feat_array_var = [Variable(att_feat.data, volatile=True) for att_feat in att_feat_array]

    greedy_search_res = model.sample(fc_feat_array_var,
                                     att_feat_array_var
                                     )

    greedy_res = greedy_search_res[0]
    rewards = compute_reward(idx_to_word, gen_result, greedy_res, data, opt)

    return rewards


def get_self_critical_reward(idx_to_word, model, fc_feats, att_feats, data, gen_result, opt):
    # get greedy decoding baseline
    greedy_search_res = model.sample(Variable(fc_feats.data, volatile=True),
                                     Variable(att_feats.data, volatile=True)
                                     )
    greedy_res = greedy_search_res[0]
    rewards = compute_reward(idx_to_word, gen_result, greedy_res, data, opt)

    return rewards
