nohup python3 -u eval_ensemble.py \
  --beam_size 1 \
  --feature_type feat_array \
  --print_beam_candidate 1 \
  --eval_split test \
  --eval_flip_ensemble 0 \
  --eval_num_models_per_gpu 2 \
  --eval_ensemble_multi_gpu 1 \
  --caption_model recurrent_fusion_model > log/eval_greedy_recurrent_fusion_model_test &

