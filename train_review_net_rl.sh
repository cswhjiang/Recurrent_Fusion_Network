seed=124
num_gpu=8

MODEL='review_net'
FEAT='resnet'
LOAD_MODEL=${MODEL}_${FEAT}

for i in {0..7}
do
  seed=$(( $seed + 1 ))
  gpu_id=$((i % $num_gpu))

  export CUDA_VISIBLE_DEVICES=${gpu_id};
  ID=${MODEL}_rl_${FEAT_MASK}_${FEAT}_${seed}
  nohup python3.6 -u main_rl.py \
    --id ${ID} \
    --caption_model ${MODEL} \
    --feature_type ${FEAT} \
    --seed ${seed} \
    --optim_lr 5e-5 \
    --checkpoint_path checkpoint_rl \
    --start_from checkpoint \
    --load_model_id ${LOAD_MODEL}_${seed}_0-best \
    --online_training 0 \
    --use_flip 1 \
    --use_crop 1 \
    --use_label_smoothing 1 \
    --learning_rate_decay_start -1 \
    --scheduled_sampling_start -1 \
    --save_checkpoint_every 5000 \
    --num_eval_no_improve 20 \
    --val_images_use 5000  > log/log_${ID} &
done
