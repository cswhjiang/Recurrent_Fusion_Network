#!/usr/bin/env bash
seed=124
num_gpu=8

MODEL='recurrent_fusion_model'
FEAT='feat_array'

for i in {0..7}
do
  seed=$(( $seed + 1 ))
  gpu_id=$((i % $num_gpu))

  export CUDA_VISIBLE_DEVICES=${gpu_id};
  ID=${MODEL}_crop_${FEAT}_${seed}
  nohup python3.6 -u main.py \
    --id ${ID} \
    --caption_model ${MODEL} \
    --feature_type ${FEAT} \
    --seed ${seed} \
    --optim_lr 5e-4 \
    --use_flip 1 \
    --use_crop 1 \
    --use_label_smoothing 1 \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --drop_prob_lm 0.3 \
    --save_checkpoint_every 5000 \
    --num_eval_no_improve 20 \
    --val_images_use 5000  > log/log_${ID} &
done
