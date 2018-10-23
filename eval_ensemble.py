import json
import numpy as np

import time
import os
import pickle

import opts
import models
from dataloader import *
# from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch
import math

import opts

# python3.6 eval_ensemble.py  --beam_size 1 --feature_type multi_feat_2 --print_beam_candidate 1 --eval_split val --eval_flip_ensemble 1
# python3.6 eval_ensemble.py  --beam_size 3 --feature_type resnet --print_beam_candidate 0 --eval_split test --eval_flip_ensemble 0
# python3.6 eval_ensemble.py  --beam_size 1 --feature_type feat_array --print_beam_candidate 1 --eval_split test --eval_flip_ensemble 0
# python3.6 eval_ensemble.py  --beam_size 1 --feature_type feat_array --print_beam_candidate 1 --eval_split test --eval_flip_ensemble 0 --eval_ensemble_multi_gpu 1 --eval_num_models_per_gpu 2
# nohup python3.6 -u eval_ensemble.py  --beam_size 1 --feature_type feat_array --print_beam_candidate 1 --eval_split test --eval_flip_ensemble 0 --eval_ensemble_multi_gpu 1 --eval_num_models_per_gpu 6 --feat_mask 1111101 --caption_model review_net_feat_array_ensemble_24 > log_new/eval_greedy_review_net_feat_array_ensemble_24_rl_1111101_feat_array_24_test &
opt = opts.parse_opt()

model_ids = []


model_ids.append('recurrent_fusion_model_crop_rl_feat_array_101')
model_ids.append('recurrent_fusion_model_crop_rl_feat_array_102')
model_ids.append('recurrent_fusion_model_crop_rl_feat_array_103')
model_ids.append('recurrent_fusion_model_crop_rl_feat_array_104')
model_ids.append('recurrent_fusion_model_crop_rl_feat_array_105')
model_ids.append('recurrent_fusion_model_crop_rl_feat_array_106')
model_ids.append('recurrent_fusion_model_crop_rl_feat_array_107')
model_ids.append('recurrent_fusion_model_crop_rl_feat_array_108')
print(model_ids)

model_name = []
model_info = []
if 'rl' in model_ids[0]:
    for model_id in model_ids:
        model_name.append('checkpoint_rl/rl_model_' + model_id + '_0-best.pth')
        model_info.append('checkpoint_rl/rl_infos_' + model_id + '_0-best.pkl')
else:
    for model_id in model_ids:
        model_name.append('checkpoint/model_' + model_id + '_0-best.pth')
        model_info.append('checkpoint/infos_' + model_id + '_0-best.pkl')

num_model = len(model_ids)
print('ensemble with ' + str(num_model) + ' models')

# Load infos
opt.infos_path = model_info[0]  # we assume that all model have the same info
with open(opt.infos_path, 'rb') as f:
    infos = pickle.load(f)

# override and collect parameters
if len(opt.input_fc_dir) == 0:
    opt.input_fc_dir = infos['opt'].input_fc_dir
    opt.input_att_dir = infos['opt'].input_att_dir
    opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
ignore = ["id", "batch_size", "beam_size", "start_from", 'print_beam_candidate', 'online_training',
          'use_official_split', 'use_flip', 'use_crop', 'language_eval', 'input_fc_flip_dir_1', 'input_att_flip_dir_1',
          'input_fc_flip_dir_2', 'input_att_flip_dir_2', 'input_fc_crop_dir_2', 'input_att_crop_dir_2',
          'input_fc_flip_crop_dir_2', 'input_att_flip_crop_dir_2', 'input_fc_crop_dir_4', 'input_att_crop_dir_4',
          'input_fc_flip_crop_dir_4', 'input_att_flip_crop_dir_4', 'official_test_id_file', 'official_val_id_file',
          'input_fc_dir_1', 'input_att_dir_1', 'input_label_h5', 'num_eval_no_improve', 'learning_rate_decay_start',
          'checkpoint_path', 'load_model_id', 'seed', 'use_label_smoothing', 'drop_prob_lm', 'scheduled_sampling_start',
          'infos_path', 'optim_lr', 'feat_array_info']

infos['opt'].eval_split = opt.eval_split
infos['opt'].beam_size = opt.beam_size
infos['opt'].eval_ensemble_multi_gpu = opt.eval_ensemble_multi_gpu
infos['opt'].eval_num_models_per_gpu = opt.eval_num_models_per_gpu
for k in vars(infos['opt']).keys():
    if k not in ignore:
        if k in vars(opt):
            print(k)
            print(vars(opt)[k])
            print(vars(infos['opt'])[k])
            # assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

opt_dict = vars(opt)
for k, v in opt_dict.items():
    print(k + ': \t' + str(v))

vocab = infos['vocab']  # ix -> word mapping

num_gpu = torch.cuda.device_count()
model_id_2_gpu_id = {}
gpu_id_2_model_id = {}
if opt.eval_ensemble_multi_gpu:
    assert (num_gpu * opt.eval_num_models_per_gpu >= num_model)
    for i in range(num_model):
        gpu_id = math.floor(i / opt.eval_num_models_per_gpu)
        model_id_2_gpu_id[i] = gpu_id
        gpu_id_2_model_id[gpu_id] = i

print(model_id_2_gpu_id)

# Setup the model
model_list = []
for i in range(num_model):
    print('loading ' + model_name[i] + ' ...')
    model = models.setup(opt)
    model.load_state_dict(torch.load(model_name[i]))
    if opt.eval_ensemble_multi_gpu:
        gpu_id = model_id_2_gpu_id[i]
        model.cuda(gpu_id)
    else:
        model.cuda()
    model.eval()
    model_list.append(model)

# Create the Data Loader instance
if len(opt.image_folder) == 0:
    loader = DataLoader(opt)
else:
    raise NotImplementedError
    # loader = DataLoaderRaw({'folder_path': opt.image_folder,
    #                         'coco_json': opt.coco_json,
    #                         'batch_size': opt.batch_size})
    # loader.ix_to_word = infos['vocab']

# Set sample options
# opt.beam_size = 3
if opt.beam_size > 1:
    opt.use_flip = 0
    opt.use_flip_or_origin = 1
    loader.use_flip = 0
    loader.aug_type = 0
    loss_1, split_predictions_1, lang_stats_1 = eval_utils.eval_ensemble(model_list, loader, vars(opt))

    if opt.eval_flip_ensemble:  # not working for now
        opt.use_flip_or_origin = 0
        loader.use_flip = 0
        loader.aug_type = 1
        loss_2, split_predictions_2, lang_stats_2 = eval_utils.eval_ensemble(model_list, loader, vars(opt))

else:
    opt.use_flip = 0
    opt.use_flip_or_origin = 1
    loader.use_flip = 0
    loader.aug_type = 0
    loss_1, split_predictions_1, lang_stats_1 = eval_utils.eval_ensemble_greedy(model_list, loader, vars(opt),
                                                                                model_id_2_gpu_id, gpu_id_2_model_id)

    if opt.eval_flip_ensemble:
        opt.use_flip_or_origin = 0
        loader.use_flip = 0
        loader.aug_type = 1
        loss_2, split_predictions_2, lang_stats_2 = eval_utils.eval_ensemble_greedy(model_list, loader, vars(opt))

# combine
if opt.eval_flip_ensemble:
    pred_1 = {}
    pred_2 = {}
    for entry in split_predictions_1:
        image_id = entry['image_id']
        pred_1[image_id] = entry
    for entry in split_predictions_2:
        image_id = entry['image_id']
        pred_2[image_id] = entry

    pred_combine = []
    for image_id in pred_1.keys():
        prob_1 = pred_1[image_id]['log_prob']
        prob_2 = pred_2[image_id]['log_prob']
        entry = {}
        entry['image_id'] = image_id
        if prob_1 > prob_2:
            entry['caption'] = pred_1[image_id]['caption']
        else:
            entry['caption'] = pred_2[image_id]['caption']
        pred_combine.append(entry)

    if opt.language_eval == 1:
        lang_stats = eval_utils.language_eval('coco', pred_combine, 'ensemble_' + opt.caption_model, opt.eval_split)

# print('loss: ', loss)
if opt.language_eval == 1:
    print(lang_stats_1)
    if opt.eval_flip_ensemble:
        print(lang_stats_2)
        print(lang_stats)
