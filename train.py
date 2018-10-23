# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np

import time
import os
import sys
from six.moves import cPickle

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils


def train(rank, model, opt, optimizer=None):
    torch.manual_seed(opt.seed + rank)
    if opt.use_cuda:
        torch.cuda.manual_seed(opt.seed + rank)

    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    infos = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_' + opt.load_model_id + '.pkl'), 'rb') as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[
                    checkme], "Command line argument and saved model disagree on '%s' " % checkme

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    lr_history = infos.get('lr_history', {})
    ss_prob_history = infos.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_image_id = infos.get('split_image_id', loader.split_image_id)
    best_val_score = 0
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    update_lr_flag = True
    if opt.caption_model == 'show_tell':
        crit = utils.LanguageModelCriterion(opt)

    elif opt.caption_model == 'review_net':
        crit = utils.ReviewNetCriterion(opt)

    elif opt.caption_model == 'recurrent_fusion_model':
        crit = utils.ReviewNetEnsembleCriterion(opt)

    else:
        raise Exception("caption_model not supported: {}".format(opt.caption_model))

    if optimizer is None:
        if opt.optim == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=opt.optim_lr,
                                   betas=(opt.optim_adam_beta1, opt.optim_adam_beta2), weight_decay=opt.optim_weight_decay)
        elif opt.optim == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=opt.optim_lr, momentum=opt.optim_momentum,
                                      alpha=opt.optim_rmsprop_alpha, weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=opt.optim_lr, momentum=opt.optim_momentum, weight_decay=opt.optim_weight_decay)
        elif opt.optim == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=opt.optim_lr, lr_decay=opt.optim_lr_decay,
                                      weight_decay=opt.optim_weight_decay)
        elif opt.optim == 'adadelta':
            optimizer = optim.Adadelta(model.parameters(), rho=opt.optim_rho, eps=opt.optim_epsilon, lr=opt.optim_lr,
                                       weight_decay=opt.optim_weight_decay)
        else:
            raise Exception("optim not supported: {}".format(opt.feature_type))

        # Load the optimizer
        if vars(opt).get('start_from', None) is not None:
            optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer_' + opt.load_model_id + '.pth')))

    num_period_best = 0
    current_score = 0
    while True:
        if update_lr_flag:
                # Assign the learning rate
            if epoch > opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate ** frac
                opt.current_lr = opt.optim_lr * decay_factor
                utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
            else:
                opt.current_lr = opt.optim_lr
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob
            update_lr_flag = False

        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')

        if opt.use_cuda:
            torch.cuda.synchronize()

        if opt.feature_type == 'feat_array':
            fc_feat_array = data['fc_feats_array']
            att_feat_array = data['att_feats_array']
            assert(len(fc_feat_array) == len(att_feat_array))
            for feat_id in range(len(fc_feat_array)):
                if opt.use_cuda:
                    fc_feat_array[feat_id] = Variable(torch.from_numpy(fc_feat_array[feat_id]), requires_grad=False).cuda()
                    att_feat_array[feat_id] = Variable(torch.from_numpy(att_feat_array[feat_id]), requires_grad=False).cuda()
                else:
                    fc_feat_array[feat_id] = Variable(torch.from_numpy(fc_feat_array[feat_id]), requires_grad=False)
                    att_feat_array[feat_id] = Variable(torch.from_numpy(att_feat_array[feat_id]), requires_grad=False)

            tmp = [data['labels'], data['masks'], data['top_words']]
            if opt.use_cuda:
                tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
            else:
                tmp = [Variable(torch.from_numpy(_), requires_grad=False) for _ in tmp]
            labels, masks, top_words = tmp

        else:
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['top_words']]
            if opt.use_cuda:
                tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
            else:
                tmp = [Variable(torch.from_numpy(_), requires_grad=False) for _ in tmp]
            fc_feats, att_feats, labels, masks, top_words = tmp

        optimizer.zero_grad()

        if opt.caption_model == 'show_tell':
            log_prob = model(fc_feats, att_feats, labels)  # (80L, 16L, 9488L)
            loss = crit(log_prob, labels[:, 1:], masks[:, 1:])

        elif opt.caption_model == 'review_net':
            log_prob, top_pred = model(fc_feats, att_feats, labels)  # (80L, 16L, 9488L)
            loss = crit(log_prob, labels[:, 1:], masks[:, 1:], top_pred, top_words, opt.reason_weight)

        elif opt.caption_model == 'recurrent_fusion_model':
            log_prob, top_pred = model(fc_feat_array, att_feat_array, labels)  # (80L, 16L, 9488L)
            loss = crit(log_prob, labels[:, 1:], masks[:, 1:], top_pred, top_words, opt.reason_weight)

        else:
            raise Exception("caption_model not supported: {}".format(opt.caption_model))

        loss.backward()

        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        train_loss = loss.data[0]
        if opt.use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if iteration % opt.losses_log_every == 0:
            loss_history[iteration] = train_loss
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if iteration % opt.save_checkpoint_every == 0:
            # eval model
            eval_kwargs = {'eval_split': 'val',
                           'dataset': opt.input_json,
                           'caption_model': opt.caption_model,
                           'reason_weight': opt.reason_weight,
                           'guiding_l1_penality': opt.guiding_l1_penality,
                           'use_cuda': opt.use_cuda,
                           'feature_type': opt.feature_type,
                           'rank': rank,
                           'val_images_use': opt.val_images_use,
                           'language_eval': 1
                           }
            eval_kwargs.update(vars(opt))
            eval_kwargs['eval_split'] = 'val'
            val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)

            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False

            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True
                num_period_best = 1
            else:
                num_period_best = num_period_best + 1

            # Dump miscalleous informations
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_image_id'] = loader.split_image_id
            infos['best_val_score'] = best_val_score
            infos['opt'] = opt
            infos['val_result_history'] = val_result_history
            infos['loss_history'] = loss_history
            infos['lr_history'] = lr_history
            infos['ss_prob_history'] = ss_prob_history
            infos['vocab'] = loader.get_vocab()
            with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '_' + str(rank) + '.pkl'), 'wb') as f:
                cPickle.dump(infos, f)

            if best_flag:
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model_' + opt.id + '_' + str(rank) + '-best.pth')
                torch.save(model.state_dict(), checkpoint_path)
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer_' + opt.id + '_' + str(rank) + '-best.pth')
                torch.save(optimizer.state_dict(), optimizer_path)
                print("model saved to {}".format(checkpoint_path))
                with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '_' + str(rank) + '-best.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
            
            if num_period_best >= opt.num_eval_no_improve:
                print('no improvement, exit')
                sys.exit()

        print("rank {}, iter {}, (epoch {}), train loss: {}, learning rate: {}, current cider: {:.3f}, best cider: {:.3f}, time: {:.3f}"
              .format(rank, iteration, epoch, train_loss, opt.current_lr, current_score, best_val_score, (end - start)))
        iteration += 1         
        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break
