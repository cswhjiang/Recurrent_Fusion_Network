# Recurrent Fusion Network for Image Captioning

This repository includes the implementations for [Recurrent Fusion Network for Image Captioning](https://arxiv.org/abs/1807.09986). 

## Requirements
- Python 3.6
- PyTorch 0.3.1
- Java

## Training
### 0. Feature extraction
All scripts for feature extraction are included in ```data/feature_extraction```. Please generate flipped and cropped images to perform data augmentation, download pre-trained models and extract features with the scripts. All extracted featuers should be put in the ```data``` directory.

### 1. Train with cross entropy loss
```bash train_recurrent_fusion_model.sh```

### 2. Training with reinforcement learning
```bash train_recurrent_fusion_model_rl.sh```

## Evaluation
Evaluate with ```eval_single.sh``` and ```eval_ensemble.sh``` to obtain metric scores for single model and ensemble of multiple models, respectively.

## Reference
If you find this repo useful, please consider citing:

```
@InProceedings{Jiang_2018_ECCV,
author = {Jiang, Wenhao and Ma, Lin and Jiang, Yu-Gang and Liu, Wei and Zhang, Tong},
title = {Recurrent Fusion Network for Image Captioning},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}
```

## Acknowledgements
Our code is based on [Ruotian Luo's implementation](https://github.com/ruotianluo/self-critical.pytorch) ans is reorganized by [Zhiming Ma](https://github.com/mazm13).


