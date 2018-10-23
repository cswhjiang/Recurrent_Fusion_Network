import models
from dataloader import *
import eval_utils
import misc.utils as utils
import pickle
import torch
import opts

opt = opts.parse_opt()
opt_dict = vars(opt)
for k, v in opt_dict.items():
    print(k + ': \t' + str(v))

# Load infos
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

ignore = ["id", "batch_size", "start_from", 'print_beam_candidate', 'online_training',
          'use_official_split', 'use_flip', 'use_crop', 'language_eval', 'input_fc_flip_dir_1', 'input_att_flip_dir_1',
          'input_fc_flip_dir_2', 'input_att_flip_dir_2', 'input_fc_crop_dir_2', 'input_att_crop_dir_2',
          'input_fc_flip_crop_dir_2', 'input_att_flip_crop_dir_2', 'input_fc_crop_dir_4', 'input_att_crop_dir_4',
          'input_fc_flip_crop_dir_4', 'input_att_flip_crop_dir_4', 'official_test_id_file', 'official_val_id_file',
          'input_fc_dir_1', 'input_att_dir_1', 'input_label_h5', 'use_label_smoothing', 'input_fc_flip_dir_4',
          'input_att_flip_dir_4', 'caption_model', 'drop_prob_lm', 'num_eval_no_improve', 'learning_rate_decay_start',
          'scheduled_sampling_start', 'seed', 'checkpoint_path', 'load_model_id', 'model_path', 'infos_path', 'optim_lr',
          'drop_prob_connect', 'val_images_use', 'reason_weight']

infos['opt'].beam_size = opt.beam_size
infos['opt'].eval_split = opt.eval_split
for k in vars(infos['opt']).keys():
    if k not in ignore:
        if k in vars(opt):
            print(k)
            print(vars(opt)[k])
            print(vars(infos['opt'])[k])
            # assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

vocab = infos['vocab']  # ix -> word mapping

# Setup the model
model = models.setup(opt)
print('loading ' + opt.model_path)
model.load_state_dict(torch.load(opt.model_path))
model.cuda()
model.eval()
# crit = utils.LanguageModelCriterion()

# opt.reason_weight = 0
# opt.use_label_smoothing = 0
if opt.caption_model == 'show_tell':
    crit = utils.LanguageModelCriterion(opt)

elif opt.caption_model == 'review_net':
    crit = utils.ReviewNetCriterion(opt)

elif opt.caption_model == 'recurrent_fusion_model':
    crit = utils.ReviewNetEnsembleCriterion(opt)

else:
    raise Exception("caption_model not supported: {}".format(opt.caption_model))

loader = DataLoader(opt)
eval_kwargs = {'eval_split': opt.eval_split,
               'beam_size': opt.beam_size,
               'dataset': opt.input_json,
               'caption_model': opt.caption_model,
               'reason_weight': opt.reason_weight,  ##
               'guiding_l1_penality': opt.guiding_l1_penality,
               'use_cuda': opt.use_cuda,
               'feature_type': opt.feature_type,
               'language_eval': opt.language_eval,
               'val_images_use': opt.val_images_use,
               'verbose': opt.verbose,
               'sample_max': opt.sample_max,
               'print_beam_candidate': opt.print_beam_candidate,
               'id': opt.id,
               'print_top_words': 0
               }


# Set sample options
# loss, split_predictions, lang_stats = eval_utils.eval_eval(model, crit, loader, vars(opt))
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)

print('loss: ', loss)
if lang_stats:
    print(lang_stats)
