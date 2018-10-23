#!/usr/bin/env bash
MODEL='recurrent_fusion_model'
FEAT='feat_array'

seed=100
gpu_id=0

export CUDA_VISIBLE_DEVICES=${gpu_id};
ID=${MODEL}_crop_${FEAT}_${seed}_single
python3.6 -u main.py \
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
  --val_images_use 5000


