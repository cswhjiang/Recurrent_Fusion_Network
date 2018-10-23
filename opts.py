# -*- coding: utf-8 -*-
import argparse
import sys
import feat_array


def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--input_json', type=str, default='data/cocotalk.json',
                        help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_label_h5', type=str, default='data/cocotalk_label.h5',
                        help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--start_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                            'infos.pkl'         : configuration;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                            """)
    parser.add_argument('--top_words_path', type=str, default='data/vocab_train.pkl',
                        help='path to top 1000 words')
    parser.add_argument('--top_words_count', type=int, default=1000,
                        help='number of top words')
    parser.add_argument('--feature_type', type=str, default='inception_v3',
                        help='type of feature: inception_v3 |  inception_v4 | resnet | densenet | feat_array')
    parser.add_argument('--official_train_id_file', type=str, default='data/official_split/official_train_id.txt',
                        help='path of official split')
    parser.add_argument('--official_val_id_file', type=str, default='data/official_split/official_val_id.txt',
                        help='path of official split')
    parser.add_argument('--official_test_id_file', type=str, default='data/official_split/official_test_id.txt',
                        help='path of official split')
    parser.add_argument('--use_official_split', type=int, default=0,
                        help='whether use official split, default is false (use Karpathy\'s split).')

    parser.add_argument('--use_flip', type=int, default=0,
                        help='use all augmentation? 0 means use origin only, 1 means use all augmentation')
    parser.add_argument('--use_crop', type=int, default=0,
                        help='whether use crop features. 0 means use only one kind of features, 1 means origin and crop')
    parser.add_argument('--aug_type', type=int, default=0,
                        help='0: origin, 1: flip, 2: crop, 3: flip-crop, only active if use_flip is 1')

    parser.add_argument('--use_label_smoothing', type=int, default=0,
                        help='whether use label smoothing')
    parser.add_argument('--label_smoothing_epsilon', type=float, default=0.1,
                        help='parameter for label smoothing')

    parser.add_argument('--use_mos', type=int, default=0,
                        help='whether use mixture of softmax')
    parser.add_argument('--num_expert', type=int, default=10,
                        help='number of experts for mixture of softmax')
    # Model settings
    parser.add_argument('--caption_model', type=str, default="show_tell",
                        help='show_tell, review_net, review_net_feat_array_ensemble_24')
    parser.add_argument('--rnn_size', type=int, default=512,
                        help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                        help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_hid_size', type=int, default=512,
                        help='the hidden size of the attention MLP; '
                             'only useful in show_attend_tell; 0 if not using hidden layer')

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=-1,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=1,  # 5.,
                        help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.0,
                        help='strength of dropout in the Language Model RNN')
    parser.add_argument('--drop_prob_reason', type=float, default=0.0,
                        help='strength of dropout in the review step')
    parser.add_argument('--drop_prob_fusion', type=float, default=0.0,
                        help='strength of dropout in the fusion step')
    parser.add_argument('--drop_prob_obj_att', type=float, default=0.0,
                        help='strength of dropout for bottom up model (obj attention lstm)')
    parser.add_argument('--drop_prob_connect', type=float, default=0.0,
                        help='strength of dropout connection to review step)')
    parser.add_argument('--seq_per_img', type=int, default=5,
                        help='number of captions to sample for each image during training. '
                             'Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
    parser.add_argument('--beam_size', type=int, default=1,
                        help='used when sample_max = 1, indicates number of beams in beam search. '
                             'Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime '
                             'but a bit worse performance.')
    parser.add_argument('--num_eval_no_improve', type=int, default=10,
                        help='exit when number of evaluations no improvement')

    # Optimization: for the Language Model
    # adam lr: 1e-4, grad_clip 1, faster
    # adagrad lr 1e-1, grad_clip 5 or 1
    parser.add_argument('--optim', type=str, default='adam',
                        help='what update to use? rmsprop | sgd | adagrad | adam | adadelta')
    parser.add_argument('--optim_lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--optim_rl_lr', type=float, default=5e-5,
                        help='learning rate for RL')
    parser.add_argument('--optim_rl_lr_ratio', type=float, default=2.0,
                        help='learning rate for RL')
    parser.add_argument('--load_lr', type=int, default=0,
                        help='whether load LR from history')
    parser.add_argument('--learning_rate_decay_start', type=int, default=1,
                        help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3,
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8,
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_adam_beta1', type=float, default=0.9,
                        help='alpha for adam')
    parser.add_argument('--optim_adam_beta2', type=float, default=0.999,
                        help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                        help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--optim_weight_decay', type=float, default=0.00001,
                        help='weight_decay')
    parser.add_argument('--optim_rmsprop_alpha', type=float, default=0.99,
                        help='alpha used for rmsprop')
    parser.add_argument('--optim_momentum', type=float, default=0.0,
                        help='used for rmsprop, sgd')
    parser.add_argument('--optim_lr_decay', type=float, default=0.0,
                        help='used for adagrad')
    parser.add_argument('--optim_rho', type=float, default=0.9,
                        help='rho used for adadelta')

    parser.add_argument('--use_ppo', type=int, default=0,
                        help='use PPO in RL?')
    parser.add_argument('--ppo_clip', type=float, default=0.2,
                        help='clip parameter for PPO')
    parser.add_argument('--ppo_k', type=int, default=10,
                        help='epoch parameter for PPO')

    parser.add_argument('--entropy_reg', type=float, default=0.01,
                        help='regularization for entropy of policy')

    parser.add_argument('--scheduled_sampling_start', type=int, default=-1,
                        help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5,
                        help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                        help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                        help='Maximum scheduled sampling prob.')

    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=5000,
                        help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=5000,
                        help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint',
                        help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=1,
                        help='Evaluate language as well (1 = yes, 0 = no)? '
                             'BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=25,
                        help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--load_best_score', type=int, default=1,
                        help='Do we load previous best score when resuming training.')

    # misc
    parser.add_argument('--id', type=str, default='',
                        help='an id identifying this run/job. '
                             'used in cross-val and appended when writing progress files')
    parser.add_argument('--load_model_id', type=str, default='',
                        help=' the id of model will be loaded')
    parser.add_argument('--train_only', type=int, default=0,
                        help='if true then use 80k, else use 110k')
    parser.add_argument('--verbose', type=int, default=0,
                        help='verbose?')
    parser.add_argument('--online_training', type=int, default=0,
                        help='if true then use 115K for training')
    parser.add_argument('--use_cuda', type=int, default=1,  # not used for now
                        help='use_coda?')
    parser.add_argument('--seed', type=int, default=100,
                        help='seed')
    parser.add_argument('--maxout', type=int, default=0,
                        help='maxout in soft attention in decoder step?')
    parser.add_argument('--review_maxout', type=int, default=0,
                        help='maxout in soft attention in review step?')
    parser.add_argument('--fusion_maxout', type=int, default=0,
                        help='maxout in soft attention in fusion step?')
    parser.add_argument('--async_opt', type=int, default=0,
                        help='asynchronous optimization, like a3c')
    parser.add_argument('--num_processes', type=int, default=4,
                        help='num_processes')
    parser.add_argument('--use_baseline', type=int, default=1,
                        help='use baseline?')

    parser.add_argument('--bleu4_weight', type=float, default=0.0,
                        help='weight for bleu4?')
    parser.add_argument('--cider_weight', type=float, default=1.0,
                        help='weight for cider?')
    parser.add_argument('--spice_weight', type=float, default=0.0,
                        help='weight for spice?')

    # self attention
    parser.add_argument('--num_head', type=int, default=8,
                        help='number of head for self-attention')
    parser.add_argument('--drop_prob_self_attn', type=float, default=0.1,
                        help='dropout rate for self-attention')

    # review net
    parser.add_argument('--num_review_steps', type=int, default=8,
                        help='number of review steps for review net')
    parser.add_argument('--num_review_steps_0', type=int, default=8,
                        help='number of review steps for fusion net')
    parser.add_argument('--reason_weight', type=float, default=1.0,
                        help='weight for discriminative training')
    parser.add_argument('--guiding_weight', type=float, default=1.0,  # 0.1?
                        help='weight for guiding network')
    parser.add_argument('--guiding_l1_penality', type=float, default=0.001,
                        help='for l1 penality in guiding network')
    parser.add_argument('--review_net_same_rnn', type=int, default=0,
                        help='use the same rnn, default is False')

    # for eval
    parser.add_argument('--eval_split', type=str, default='test',
                        help='eval | test')
    parser.add_argument('--eval_flip_ensemble', type=int, default=0,
                        help='use flip to ensemble?')

    # For evaluation on a folder of images:
    parser.add_argument('--image_folder', type=str, default='',
                        help='If this is nonempty then will predict on the images in this folder path')
    parser.add_argument('--image_root', type=str, default='',
                        help='In case the image paths have to be preprended with a root path to an image folder')
    parser.add_argument('--model_path', type=str, default='',
                        help='path to model to evaluate')
    parser.add_argument('--infos_path', type=str, default='',
                        help='path to infos to evaluate')
    parser.add_argument('--sample_max', type=int, default=1,
                        help='1 = sample argmax words. 0 = sample from distributions.')
    parser.add_argument('--print_beam_candidate', type=int, default=0,
                        help='print beam candidates?')
    parser.add_argument('--eval_ensemble_multi_gpu', type=int, default=0,
                        help='use multi-gpu to ensemble?')
    parser.add_argument('--eval_num_models_per_gpu', type=int, default=4,
                        help='how many models on one gpu?')

    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert 0 <= args.drop_prob_lm <= 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"

    args.feat_array_info = []
    if args.feature_type == 'inception_v3':
        feat_info = feat_array.inception_v3_info

    elif args.feature_type == 'inception_v4':
        feat_info = feat_array.inception_v4_info

    elif args.feature_type == 'resnet':
        feat_info = feat_array.resnet_info

    elif args.feature_type == 'densenet':
        feat_info = feat_array.densenet_info

    elif args.feature_type == 'inception_resnet_v2':
        feat_info = feat_array.inception_resnet_v2_info

    elif args.feature_type == 'feat_array':
        feat_info = None
        args.feat_array_info = feat_array.feat_array_info
    else:
        raise Exception("feature_type not supported: {}".format(opt.feature_type))

    if feat_info is not None:
        args.input_fc_dir = feat_info['original']['fc']
        args.input_att_dir = feat_info['original']['att']

        args.input_fc_flip_dir = feat_info['flip']['fc']
        args.input_att_flip_dir = feat_info['flip']['att']

        args.input_fc_crop_dir = feat_info['crop_tr']['fc']
        args.input_att_crop_dir = feat_info['crop_tr']['att']

        args.input_fc_flip_crop_dir = feat_info['flip_crop_tr']['fc']
        args.input_att_flip_crop_dir = feat_info['flip_crop_tr']['att']

        args.input_fc_crop_tl_dir = feat_info['crop_tl']['fc']
        args.input_fc_crop_bl_dir = feat_info['crop_bl']['fc']
        args.input_fc_crop_br_dir = feat_info['crop_br']['fc']

        args.input_att_crop_tl_dir = feat_info['crop_tl']['att']
        args.input_att_crop_bl_dir = feat_info['crop_bl']['att']
        args.input_att_crop_br_dir = feat_info['crop_br']['att']

        args.input_fc_flip_crop_tl_dir = feat_info['flip_crop_tl']['fc']
        args.input_fc_flip_crop_bl_dir = feat_info['flip_crop_bl']['fc']
        args.input_fc_flip_crop_br_dir = feat_info['flip_crop_br']['fc']

        args.input_att_flip_crop_tl_dir = feat_info['flip_crop_tl']['att']
        args.input_att_flip_crop_bl_dir = feat_info['flip_crop_bl']['att']
        args.input_att_flip_crop_br_dir = feat_info['flip_crop_br']['att']

        args.fc_feat_size = feat_info['fc_feat_size']
        args.att_feat_size = feat_info['att_feat_size']
        args.att_num = feat_info['att_num']

    return args


if __name__ == '__main__':
    opt = parse_opt()
    opt_dict = vars(opt)
    for k, v in opt_dict.items():
        print(k + ': \t' + str(v))
    print(opt.feat_array_info)
