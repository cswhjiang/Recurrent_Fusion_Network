#!/usr/bin/env bash
num_gpu=8
MODEL='inception_v4'
data_dir='/data1/ailab_view/mightma/image_caption_for_chinese/data/'
dirs=($(ls ${data_dir}ai_challenger/ ))
dir_size=${#dirs[@]}

for i in $(seq 0 $((${dir_size}-1)))
do
    echo ${i}
    echo $((i % $num_gpu))
    echo ${data_dir}ai_challenger/${dirs[$i]}
    echo ${data_dir}ai_challenger_feature/$feat_${MODEL}
    echo ai_${MODEL}_fc
    echo ai_${MODEL}_att
    echo ${dirs[$i]}.log
    echo 
    export CUDA_VISIBLE_DEVICES=$((i % $num_gpu))
    nohup python3 -u extract_feats_inception_v4.py \
        --image_path ${data_dir}ai_challenger/${dirs[$i]} \
        --out_dir ${data_dir}ai_challenger_feature/$feat_${MODEL} \
        --fc_dir ai_${MODEL}_fc_${dirs[$i]} \
        --att_dir ai_${MODEL}_att_${dirs[$i]} > ${dirs[$i]}.log &
done
