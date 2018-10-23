# -*- coding: utf-8 -*-

feat_array_info = []

# resnet
resnet_info = {
    'fc_feat_size': 2048,
    'att_feat_size': 2048,
    'att_num': 196,
    'original': {
        'fc': 'data/cocotalk_resnet_fc',
        'att': 'data/cocotalk_resnet_att'
    },
    'flip': {
        'fc': 'data/cocotalk_resnet_fc_flip',
        'att': 'data/cocotalk_resnet_att_flip'
    },
    'crop_tr': {
        'fc': 'data/cocotalk_resnet_fc_crop_tr',
        'att': 'data/cocotalk_resnet_att_crop_tr'
    },
    'crop_tl': {
        'fc': 'data/cocotalk_resnet_fc_crop_tl',
        'att': 'data/cocotalk_resnet_att_crop_tl'
    },
    'crop_bl': {
        'fc': 'data/cocotalk_resnet_fc_crop_bl',
        'att': 'data/cocotalk_resnet_att_crop_bl'
    },
    'crop_br': {
        'fc': 'data/cocotalk_resnet_fc_crop_br',
        'att': 'data/cocotalk_resnet_att_crop_br'
    },
    'flip_crop_tr': {
        'fc': 'data/cocotalk_resnet_fc_flip_crop_tr',
        'att': 'data/cocotalk_resnet_att_flip_crop_tr'
    },
    'flip_crop_tl': {
        'fc': 'data/cocotalk_resnet_fc_flip_crop_tl',
        'att': 'data/cocotalk_resnet_att_flip_crop_tl'
    },
    'flip_crop_bl': {
        'fc': 'data/cocotalk_resnet_fc_flip_crop_bl',
        'att': 'data/cocotalk_resnet_att_flip_crop_bl'
    },
    'flip_crop_br': {
        'fc': 'data/cocotalk_resnet_fc_flip_crop_br',
        'att': 'data/cocotalk_resnet_att_flip_crop_br'
    }
}

# inception v4
inception_v4_info = {
    'fc_feat_size': 1536,
    'att_feat_size': 1536,
    'att_num': 64,
    'original': {
        'fc': 'data/cocotalk_inception_v4_fc',
        'att': 'data/cocotalk_inception_v4_att'
    },
    'flip': {
        'fc': 'data/cocotalk_inception_v4_fc_flip',
        'att': 'data/cocotalk_inception_v4_att_flip'
    },
    'crop_tr': {
        'fc': 'data/cocotalk_inception_v4_fc_crop_tr',
        'att': 'data/cocotalk_inception_v4_att_crop_tr'
    },
    'crop_tl': {
        'fc': 'data/cocotalk_inception_v4_fc_crop_tl',
        'att': 'data/cocotalk_inception_v4_att_crop_tl'
    },
    'crop_bl': {
        'fc': 'data/cocotalk_inception_v4_fc_crop_bl',
        'att': 'data/cocotalk_inception_v4_att_crop_bl'
    },
    'crop_br': {
        'fc': 'data/cocotalk_inception_v4_fc_crop_br',
        'att': 'data/cocotalk_inception_v4_att_crop_br'
    },
    'flip_crop_tr': {
        'fc': 'data/cocotalk_inception_v4_fc_flip_crop_tr',
        'att': 'data/cocotalk_inception_v4_att_flip_crop_tr'
    },
    'flip_crop_tl': {
        'fc': 'data/cocotalk_inception_v4_fc_flip_crop_tl',
        'att': 'data/cocotalk_inception_v4_att_flip_crop_tl'
    },
    'flip_crop_bl': {
        'fc': 'data/cocotalk_inception_v4_fc_flip_crop_bl',
        'att': 'data/cocotalk_inception_v4_att_flip_crop_bl'
    },
    'flip_crop_br': {
        'fc': 'data/cocotalk_inception_v4_fc_flip_crop_br',
        'att': 'data/cocotalk_inception_v4_att_flip_crop_br'
    }
}

# inception v3
inception_v3_info = {
    'fc_feat_size': 2048,
    'att_feat_size': 1280,
    'att_num': 64,
    'original': {
        'fc': 'data/cocotalk_inception_v3_fc',
        'att': 'data/cocotalk_inception_v3_att'
    },
    'flip': {
        'fc': 'data/cocotalk_inception_v3_fc_flip',
        'att': 'data/cocotalk_inception_v3_att_flip'
    },
    'crop_tr': {
        'fc': 'data/cocotalk_inception_v3_fc_crop',
        'att': 'data/cocotalk_inception_v3_att_crop'
    },
    'crop_tl': {
        'fc': 'data/cocotalk_inception_v3_fc_crop_tl',
        'att': 'data/cocotalk_inception_v3_att_crop_tl'
    },
    'crop_bl': {
        'fc': 'data/cocotalk_inception_v3_fc_crop_bl',
        'att': 'data/cocotalk_inception_v3_att_crop_bl'
    },
    'crop_br': {
        'fc': 'data/cocotalk_inception_v3_fc_crop_br',
        'att': 'data/cocotalk_inception_v3_att_crop_br'
    },
    'flip_crop_tr': {
        'fc': 'data/cocotalk_inception_v3_fc_flip_crop',
        'att': 'data/cocotalk_inception_v3_att_flip_crop'
    },
    'flip_crop_tl': {
        'fc': 'data/cocotalk_inception_v3_fc_flip_crop_tl',
        'att': 'data/cocotalk_inception_v3_att_flip_crop_tl'
    },
    'flip_crop_bl': {
        'fc': 'data/cocotalk_inception_v3_fc_flip_crop_bl',
        'att': 'data/cocotalk_inception_v3_att_flip_crop_bl'
    },
    'flip_crop_br': {
        'fc': 'data/cocotalk_inception_v3_fc_flip_crop_br',
        'att': 'data/cocotalk_inception_v3_att_flip_crop_br'
    }
}

# densenet
densenet_info = {
    'fc_feat_size': 2208,
    'att_feat_size': 2208,
    'att_num': 49,
    'original': {
        'fc': 'data/cocotalk_densenet_fc',
        'att': 'data/cocotalk_densenet_att'
    },
    'flip': {
        'fc': 'data/cocotalk_densenet_fc_flip',
        'att': 'data/cocotalk_densenet_att_flip'
    },
    'crop_tr': {
        'fc': 'data/cocotalk_densenet_fc_crop_tr',
        'att': 'data/cocotalk_densenet_att_crop_tr'
    },
    'crop_tl': {
        'fc': 'data/cocotalk_densenet_fc_crop_tl',
        'att': 'data/cocotalk_densenet_att_crop_tl'
    },
    'crop_bl': {
        'fc': 'data/cocotalk_densenet_fc_crop_bl',
        'att': 'data/cocotalk_densenet_att_crop_bl'
    },
    'crop_br': {
        'fc': 'data/cocotalk_densenet_fc_crop_br',
        'att': 'data/cocotalk_densenet_att_crop_br'
    },
    'flip_crop_tr': {
        'fc': 'data/cocotalk_densenet_fc_flip_crop_tr',
        'att': 'data/cocotalk_densenet_att_flip_crop_tr'
    },
    'flip_crop_tl': {
        'fc': 'data/cocotalk_densenet_fc_flip_crop_tl',
        'att': 'data/cocotalk_densenet_att_flip_crop_tl'
    },
    'flip_crop_bl': {
        'fc': 'data/cocotalk_densenet_fc_flip_crop_bl',
        'att': 'data/cocotalk_densenet_att_flip_crop_bl'
    },
    'flip_crop_br': {
        'fc': 'data/cocotalk_densenet_fc_flip_crop_br',
        'att': 'data/cocotalk_densenet_att_flip_crop_br'
    }
}

# inception_resnet_v2
inception_resnet_v2_info = {
    'fc_feat_size': 1536,
    'att_feat_size': 1536,
    'att_num': 64,
    'original': {
        'fc': 'data/cocotalk_inception_resnet_v2_fc',
        'att': 'data/cocotalk_inception_resnet_v2_att'
    },
    'flip': {
        'fc': 'data/cocotalk_inception_resnet_v2_fc_flip',
        'att': 'data/cocotalk_inception_resnet_v2_att_flip'
    },
    'crop_tr': {
        'fc': 'data/cocotalk_inception_resnet_v2_fc_crop_tr',
        'att': 'data/cocotalk_inception_resnet_v2_att_crop_tr'
    },
    'crop_tl': {
        'fc': 'data/cocotalk_inception_resnet_v2_fc_crop_tl',
        'att': 'data/cocotalk_inception_resnet_v2_att_crop_tl'
    },
    'crop_bl': {
        'fc': 'data/cocotalk_inception_resnet_v2_fc_crop_bl',
        'att': 'data/cocotalk_inception_resnet_v2_att_crop_bl'
    },
    'crop_br': {
        'fc': 'data/cocotalk_inception_resnet_v2_fc_crop_br',
        'att': 'data/cocotalk_inception_resnet_v2_att_crop_br'
    },
    'flip_crop_tr': {
        'fc': 'data/cocotalk_inception_resnet_v2_fc_flip_crop_tr',
        'att': 'data/cocotalk_inception_resnet_v2_att_flip_crop_tr'
    },
    'flip_crop_tl': {
        'fc': 'data/cocotalk_inception_resnet_v2_fc_flip_crop_tl',
        'att': 'data/cocotalk_inception_resnet_v2_att_flip_crop_tl'
    },
    'flip_crop_bl': {
        'fc': 'data/cocotalk_inception_resnet_v2_fc_flip_crop_bl',
        'att': 'data/cocotalk_inception_resnet_v2_att_flip_crop_bl'
    },
    'flip_crop_br': {
        'fc': 'data/cocotalk_inception_resnet_v2_fc_flip_crop_br',
        'att': 'data/cocotalk_inception_resnet_v2_att_flip_crop_br'
    }
}

feat_array_info.append(resnet_info)
feat_array_info.append(inception_v4_info)
feat_array_info.append(inception_v3_info)
feat_array_info.append(densenet_info)
feat_array_info.append(inception_resnet_v2_info)
