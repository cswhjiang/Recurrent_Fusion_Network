#!/usr/bin/env bash
num_gpu=8
MODEL='inception_v3'
data_dir='/data1/ailab_view/mightma/image_caption_for_chinese/data/'
dirs=($(ls ${data_dir}ai_challenger/ ))
dir_size=${#dirs[@]}
i=0

export CUDA_VISIBLE_DEVICES=$((i % $num_gpu))
python3 -u extract_feats_inception_v3.py \
    --image_path ${data_dir}ai_challenger/${dirs[$i]} \
    --out_dir ${data_dir}ai_challenger_feature/$feat_${MODEL} \
    --fc_dir ai_${MODEL}_fc_${dirs[$i]} \
    --att_dir ai_${MODEL}_att_${dirs[$i]}

