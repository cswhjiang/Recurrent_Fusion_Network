import os
import copy

import numpy as np
import misc.utils as utils
import torch

from misc.ShowTellModel import ShowTellModel
from misc.ReviewNetModel import ReviewNetModel
from misc.RecurrentFusionModel import RecurrentFusionModel
# import torch.nn as nn


def setup(opt):

    if opt.caption_model == 'show_tell':
        model = ShowTellModel(opt)
    elif opt.caption_model == 'review_net':
        model = ReviewNetModel(opt)
    elif opt.caption_model == 'recurrent_fusion_model':
        model = RecurrentFusionModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist
        print('start_from: ' + opt.start_from)
        print('load_model_id: ' + opt.load_model_id)
        print('infos file: ' + os.path.join(opt.start_from, "infos_" + opt.load_model_id + ".pkl"))

        assert os.path.isdir(opt.start_from), " %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from, "infos_" + opt.load_model_id+".pkl")), \
            "infos.pkl file does not exist in path %s" % opt.start_from

        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model_'+opt.load_model_id+'.pth')))

    return model
