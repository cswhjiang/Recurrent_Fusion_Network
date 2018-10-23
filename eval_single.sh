#!/usr/bin/env bash

seed=126
python3 eval.py \
    --model_path checkpoint_rl/rl_model_recurrent_fusion_model_crop_feat_array_${seed}_0-best.pth \
    --infos_path checkpoint_rl/rl_infos_recurrent_fusion_model_crop_feat_array_${seed}_0-best.pkl \
    --language_eval 1 \
    --caption_model recurrent_fusion_model \
    --feature_type feat_array \
    --eval_split test

