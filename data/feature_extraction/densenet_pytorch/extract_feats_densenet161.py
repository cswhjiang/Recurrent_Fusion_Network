# -*- coding: utf-8 -*-
import os
from random import shuffle, seed

import numpy as np
import time
import argparse
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

from PIL import Image
import multiprocessing as mp
import math

import densenet as models

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


cwd = os.getcwd()
model_dir = 'models'


# python3.6  extract_feats_densenet161.py --fc_dir cocotalk_fc_crop_tl --att_dir cocotalk_att_crop_tl
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='/data1/ailab_view/wenhaojiang/data/mscoco',
                        help='dir for all images')
    parser.add_argument('--out_dir', type=str, default='/data1/ailab_view/wenhaojiang/data/feat_resnet',
                        help='base dir for output')
    parser.add_argument('--fc_dir', type=str, default='cocotalk_fc_crop_tl',
                        help='dir for fc')
    parser.add_argument('--att_dir', type=str, default='cocotalk_att_crop_tl',
                        help='dir for att')
    # parser.add_argument('--model', type=str, default='/data1/ailab_view/wenhaojiang/data/feature_extraction/resnet/model/resnet101.pth',
    #                     help='path for resnet101.pth')
    args = parser.parse_args()
    return args


def get_image_id(file_name):
    file_name = file_name.strip()
    image_name = file_name.split('.')[0]
    # image_id = int(image_name.split('_')[-1])  # coco
    # image_id = int(image_name)  # flickr30k
    image_id = image_name # ai challenger
    return image_id
  
  
def main(rank, imgs, jpg_path, out_dir ):
    img_scale = 224
    img_crop = 224

    cwd = os.getcwd()
    model_dir = 'models'
    densenet = models.densenet161(pretrained=True, model_dir=os.path.join(cwd, model_dir))
    densenet.cuda()
    densenet.eval()

    seed(123)  # make reproducible
    dir_fc = os.path.join(out_dir, opt.fc_dir)
    dir_att = os.path.join(out_dir, opt.att_dir)

    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    if not os.path.isdir(dir_att):
        os.mkdir(dir_att)

    for i, img_name in enumerate(imgs):
        t0 = time.time()
        print(img_name)
        image_id = get_image_id(img_name)

        # load the image
        img = Image.open(os.path.join(jpg_path, img_name)) # (640, 480), RGB
        img = transforms.Compose([
            transforms.Scale(img_scale),
            transforms.CenterCrop(img_crop),
            transforms.ToTensor(),
            normalize,
        ])(img)

        if img.size(0) == 1:
            img = torch.cat([img, img, img], dim=0)

        img = img.unsqueeze(0)
        # print(img.size())
        input_var = Variable(img, volatile=True)
        result = densenet.forward(input_var.cuda())
        fc = result[1].squeeze()
        att = result[2].squeeze()
        att = torch.transpose(att, 0, 2)

        # write to pkl
        np.save(os.path.join(dir_fc, str(image_id)), fc.data.cpu().float().numpy())
        np.savez_compressed(os.path.join(dir_att, str(image_id)), feat=att.data.cpu().float().numpy())

        print("{} {}  {}  time cost: {:.3f}".format(rank, i, img_name, time.time()-t0))


if __name__ == "__main__":
    opt = parse_opt()
    jpg_path = opt.image_path
    out_dir = opt.out_dir

    imgs = []
    for subdir, dirs, files in os.walk(jpg_path):
        for f in files:
            f = f.strip()
            imgs.append(f)

    # main(0, imgs, jpg_path, out_dir)

    processes = []
    num_processes = 8
    N = len(imgs)  # 164062
    C = int(math.ceil(N*1.0/num_processes))

    for rank in range(num_processes):
        cur_start = rank * C
        cur_end = (rank + 1) * C
        current_list = imgs[cur_start:cur_end]
        print('current image number: ' + str(len(current_list)))
        p = mp.Process(target=main, args=(rank, current_list, jpg_path, out_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
