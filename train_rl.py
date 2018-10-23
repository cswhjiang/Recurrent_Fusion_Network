import torch
from torch.autograd import Variable
import torch.optim as optim

import numpy as np

import time
import os
import sys
from six.moves import cPickle
import operator

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
import get_rewards


def train(rank, model, opt, optimizer=None):
    torch.manual_seed(opt.seed + rank)
    if opt.use_cuda:
        torch.cuda.manual_seed(opt.seed + rank)

    loader = DataLoader(opt)
    index_2_word = loader.get_vocab()
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    
    infos = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_' + opt.load_model_id + '.pkl'), 'rb') as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            # for checkme in need_be_same:
            #     assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    lr_history = infos.get('lr_history', {})
    ss_prob_history = infos.get('ss_prob_history', {})

    sorted_lr = sorted(lr_history.items(), key=operator.itemgetter(1))
    if opt.load_lr and len(lr_history) > 0:
        opt.optim_rl_lr = sorted_lr[0][1] / opt.optim_rl_lr_ratio

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_image_id = infos.get('split_image_id', loader.split_image_id)

    entropy_reg = opt.entropy_reg
    best_val_score = 0
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    update_lr_flag = True

    if opt.caption_model == 'show_tell':
        crit = utils.LanguageModelCriterion(opt)
        rl_crit = utils.RewardCriterion(opt)

    elif opt.caption_model == 'review_net':
        crit = utils.ReviewNetCriterion(opt)
        rl_crit = utils.ReviewNetRewardCriterion(opt)

    elif opt.caption_model == 'recurrent_fusion_model':
        crit = utils.ReviewNetEnsembleCriterion(opt)
        rl_crit = utils.ReviewNetRewardCriterion(opt)

    else:
        raise Exception("caption_model not supported: {}".format(opt.caption_model))

    if optimizer is None:
        if opt.optim == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=opt.optim_rl_lr,
                                   betas=(opt.optim_adam_beta1, opt.optim_adam_beta2), weight_decay=opt.optim_weight_decay)
        elif opt.optim == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=opt.optim_rl_lr, momentum=opt.optim_momentum,
                                      alpha=opt.optim_rmsprop_alpha, weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=opt.optim_rl_lr, momentum=opt.optim_momentum, weight_decay=opt.optim_weight_decay)
        elif opt.optim == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=opt.optim_rl_lr, lr_decay=opt.optim_lr_decay,
                                      weight_decay=opt.optim_weight_decay)
        elif opt.optim == 'adadelta':
            optimizer = optim.Adadelta(model.parameters(), rho=opt.optim_rho, eps=opt.optim_epsilon, lr=opt.optim_rl_lr,
                                       weight_decay=opt.optim_weight_decay)
        else:
            raise Exception("optim not supported: {}".format(opt.feature_type))

        # Load the optimizer
        if opt.load_lr and vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from, 'optimizer_' + opt.load_model_id + '.pth')):
            optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer_' + opt.load_model_id + '.pth')))
            utils.set_lr(optimizer, opt.optim_rl_lr)

    num_period_best = 0   
    current_score = 0
    while True:
        if update_lr_flag:
            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate ** frac
                opt.current_lr = opt.optim_rl_lr * decay_factor
                utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
            else:
                opt.current_lr = opt.optim_rl_lr
            update_lr_flag = False

        start = time.time()
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
            gen_result, sample_logprobs, logprobs_all = model.sample(fc_feats, att_feats, {'sample_max': 0})
            rewards = get_rewards.get_self_critical_reward(index_2_word, model, fc_feats, att_feats, data, gen_result, opt)
            sample_logprobs_old = Variable(sample_logprobs.data, requires_grad=False)

            if opt.use_cuda:
                loss = rl_crit(sample_logprobs, gen_result, Variable(torch.from_numpy(rewards).float().cuda(), requires_grad=False), logprobs_all, entropy_reg, sample_logprobs_old, opt)
            else:
                loss = rl_crit(sample_logprobs, gen_result, Variable(torch.from_numpy(rewards).float(), requires_grad=False), logprobs_all, entropy_reg, sample_logprobs_old, opt)

        elif opt.caption_model == 'recurrent_fusion_model':
            gen_result, sample_logprobs, logprobs_all, top_pred = model.sample(fc_feat_array, att_feat_array, {'sample_max': 0})
            rewards = get_rewards.get_self_critical_reward_feat_array(index_2_word, model, fc_feat_array, att_feat_array, data, gen_result, opt)
            sample_logprobs_old = Variable(sample_logprobs.data, requires_grad=False)

            if opt.use_cuda:
                loss = rl_crit(sample_logprobs, gen_result, Variable(torch.from_numpy(rewards).float().cuda(), requires_grad=False), logprobs_all,
                               entropy_reg, top_pred, top_words, opt.reason_weight, sample_logprobs_old, opt)
            else:
                loss = rl_crit(sample_logprobs, gen_result, Variable(torch.from_numpy(rewards).float(), requires_grad=False), logprobs_all,
                               entropy_reg, top_pred, top_words, opt.reason_weight, sample_logprobs_old, opt)

        elif opt.caption_model == 'review_net':
            gen_result, sample_logprobs, logprobs_all, top_pred = model.sample(fc_feats, att_feats, {'sample_max': 0})
            rewards = get_rewards.get_self_critical_reward(index_2_word, model, fc_feats, att_feats, data, gen_result, opt)
            sample_logprobs_old = Variable(sample_logprobs.data, requires_grad=False)

            if opt.use_cuda:
                loss = rl_crit(sample_logprobs, gen_result, Variable(torch.from_numpy(rewards).float().cuda(), requires_grad=False), logprobs_all,
                               entropy_reg, top_pred, top_words, opt.reason_weight, sample_logprobs_old, opt)
            else:
                loss = rl_crit(sample_logprobs, gen_result, Variable(torch.from_numpy(rewards).float(), requires_grad=False), logprobs_all,
                               entropy_reg, top_pred, top_words, opt.reason_weight, sample_logprobs_old, opt)

        else:
            raise Exception("caption_model not supported: {}".format(opt.caption_model))

        if opt.use_ppo and opt.ppo_k > 0:
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()

        train_loss = loss.data[0]
        if opt.use_ppo:
            for i in range(opt.ppo_k):
                print(i)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                utils.clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()

        if opt.use_cuda:
            torch.cuda.synchronize()
        end = time.time()
                
        # Update the iteration and epoch
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if iteration % opt.losses_log_every == 0:
            loss_history[iteration] = np.mean(rewards[:,0])
            lr_history[iteration] = opt.current_lr

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
                           'rank': rank
                           }
            eval_kwargs.update(vars(opt))
            eval_kwargs['eval_split'] = 'val'
            val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)

            # Write validation result into summary
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}
            print("iter {} (epoch {}), val_loss = {:.3f}" .format(iteration, epoch, val_loss))
            
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
            with open(os.path.join(opt.checkpoint_path, 'rl_infos_' + opt.id + '_' + str(rank) + '.pkl'), 'wb') as f:
                cPickle.dump(infos, f)

            if best_flag:
                checkpoint_path = os.path.join(opt.checkpoint_path, 'rl_model_' + opt.id + '_' + str(rank) + '-best.pth')
                torch.save(model.state_dict(), checkpoint_path)
                optimizer_path = os.path.join(opt.checkpoint_path, 'rl_optimizer_' + opt.id + '_' + str(rank) + '-best.pth')
                torch.save(optimizer.state_dict(), optimizer_path)
                print("model saved to {}".format(checkpoint_path))
                with open(os.path.join(opt.checkpoint_path, 'rl_infos_'+opt.id + '_' + str(rank) + '-best.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)

            if num_period_best >= opt.num_eval_no_improve:
                print('no improvement, exit')
                sys.exit() 
        print("rank {}, iter {}, (epoch {}), avg_reward: {:.3f}, train_loss: {}, learning rate: {}, current cider: {:.3f}, best cider: {:.3f}, time: {:.3f}" \
              .format(rank, iteration, epoch, np.mean(rewards[:, 0]), train_loss, opt.current_lr, current_score, best_val_score, (end-start)))
        
        iteration += 1       
        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break
#
# opt = opts.parse_opt()
# train(opt)
