seed=100
gpu_id=0

MODEL='recurrent_fusion_model'
FEAT='feat_array'
LOAD_MODEL='review_net_feat_array_ensemble_24_crop_11111000_feat_array'

export CUDA_VISIBLE_DEVICES=${gpu_id};
ID=${MODEL}_crop_rl_${FEAT_MASK}_${FEAT}_${seed}_test
python3.6 -u main_rl.py \
  --id ${ID} \
  --caption_model ${MODEL} \
  --feature_type ${FEAT} \
  --seed ${seed} \
  --checkpoint_path checkpoint_rl \
  --start_from checkpoint \
  --load_model_id ${LOAD_MODEL}_${seed}_0-best \
  --online_training 0 \
  --optim_lr 5e-5 \
  --use_flip 1 \
  --use_crop 1 \
  --learning_rate_decay_start -1 \
  --scheduled_sampling_start -1 \
  --save_checkpoint_every 5000 \
  --num_eval_no_improve 20 \
  --val_images_use 5000

