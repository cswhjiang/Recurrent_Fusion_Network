import os
from random import shuffle, seed

# non-standard dependencies:
import numpy as np
import torch
from torch.autograd import Variable
import skimage.io
import argparse
# from multiprocessing import Process

from torchvision import transforms as trn
preprocess = trn.Compose([
        # trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from misc.resnet_utils import myResnet
import misc.resnet as resnet


# python3.6  extract_resnet_feats --fc_dir cocotalk_fc_crop_tl --att_dir cocotalk_att_crop_tl
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
    parser.add_argument('--model', type=str, default='/data1/ailab_view/wenhaojiang/data/feature_extraction/resnet/model/resnet101.pth',
                        help='path for resnet101.pth')
    args = parser.parse_args()
    return args


def get_image_id(file_name):
    file_name = file_name.strip()
    image_name = file_name.split('.')[0]
    # image_id = int(image_name.split('_')[-1])  # coco
    # image_id = int(image_name)  # flickr30k
    image_id = image_name # ai challenger
    return image_id
  
  
def main(opt):
    jpg_path = opt.image_path
    out_dir = opt.out_dir
    model_path = opt.model

    net = getattr(resnet, 'resnet101')()
    net.load_state_dict(torch.load(model_path))
    my_resnet = myResnet(net)
    my_resnet.cuda()
    my_resnet.eval()

    imgs = []
    for subdir, dirs, files in os.walk(jpg_path):
        for f in files:
            f = f.strip()
            imgs.append(f)

    N = len(imgs)

    seed(123)  # make reproducible

    dir_fc = os.path.join(out_dir, opt.fc_dir)
    dir_att = os.path.join(out_dir, opt.att_dir)

    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    if not os.path.isdir(dir_att):
        os.mkdir(dir_att)

    for i, img in enumerate(imgs):
        print(img)
        image_id = get_image_id(img)
        # load the image
        I = skimage.io.imread(os.path.join(jpg_path, img))

        # handle grayscale input images
        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)

        I = I.astype('float32')/255.0
        I = torch.from_numpy(I.transpose([2, 0, 1])).cuda()
        I = Variable(preprocess(I), volatile=True)
        tmp_fc, tmp_att = my_resnet(I, 14)
        # write to pkl
        np.save(os.path.join(dir_fc, str(image_id)), tmp_fc.data.cpu().float().numpy())
        np.savez_compressed(os.path.join(dir_att, str(image_id)), feat=tmp_att.data.cpu().float().numpy())

        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
