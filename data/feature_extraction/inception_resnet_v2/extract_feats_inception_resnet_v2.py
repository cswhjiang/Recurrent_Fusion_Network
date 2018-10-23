#! encoding: UTF-8

import os
import glob
import argparse
import time
import urllib
import numpy as np
import tensorflow as tf
from nets import inception
from preprocessing import inception_preprocessing


# python extract_feats_inception_resnet_v2.py --fc_dir cocotalk_inception_resnet_v2_fc_flip --att_dir cocotalk_inception_resnet_v2_att_flip
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='/data1/ailab_view/wenhaojiang/data/mscoco_flip',
                        help='dir for all images')
    parser.add_argument('--out_dir', type=str, default='/data1/ailab_view/wenhaojiang/data/feat_inception_resnet_v2',
                        help='base dir for output')
    parser.add_argument('--fc_dir', type=str, default='cocotalk_inception_resnet_v2_fc_flip',
                        help='dir for fc')
    parser.add_argument('--att_dir', type=str, default='cocotalk_inception_resnet_v2_att_flip',
                        help='dir for att')
    parser.add_argument('--model_path', type=str, default='/data1/ailab_view/image_captioning/self-critical.pytorch-master/feature_extraction/inception_resnet_v2/checkpoints',
                        help='path for resnet101.pth')
    args = parser.parse_args()
    return args


def get_image_id(file_name):
    file_name = file_name.strip()
    image_name = file_name.split('.')[0]
    # image_id = int(image_name.split('_')[-1])  # coco
    # image_id = int(image_name)  # flickr30k
    image_id = image_name # ai chellenge
    return image_id


def main(opt):
    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config.gpu_options.allow_growth = True

    jpg_path = opt.image_path
    images_lists = []
    for subdir, dirs, files in os.walk(jpg_path):
        for f in files:
            f = f.strip()
            images_lists.append(os.path.join(jpg_path, f))

    att_dir = os.path.join(opt.out_dir, opt.att_dir)
    fc_dir = os.path.join(opt.out_dir, opt.fc_dir)

    if not tf.gfile.Exists(fc_dir):
        tf.gfile.MakeDirs(fc_dir)
    if not tf.gfile.Exists(att_dir):
        tf.gfile.MakeDirs(att_dir)

    checkpoints_dir = opt.model_path

    slim = tf.contrib.slim
    image_size = inception.inception_resnet_v2.default_image_size

    tf_image = tf.placeholder(tf.string, None)
    image = tf.image.decode_jpeg(tf_image, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images = tf.expand_dims(processed_image, 0)

    with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
        tf_feats_att, tf_feats_fc = inception.inception_resnet_v2(processed_images, num_classes=1001, is_training=False)

    init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'inception_resnet_v2_2016_08_30.ckpt'),
                                             slim.get_model_variables('InceptionResnetV2'))

    with tf.Session(config=config) as sess:
        init_fn(sess)

        for idx, image_path in enumerate(images_lists):
            image_name = os.path.basename(image_path)
            image_id = get_image_id(image_name)

            time_start = time.time()

            url = 'file://' + image_path
            image_string = urllib.request.urlopen(url).read()

            feat_conv, feat_fc = sess.run([tf_feats_att, tf_feats_fc], feed_dict={tf_image: image_string})
            feat_conv = np.squeeze(feat_conv)
            feat_fc = np.squeeze(feat_fc)

            np.save(os.path.join(fc_dir, str(image_id)), feat_fc)
            np.savez_compressed(os.path.join(att_dir, str(image_id)), feat=feat_conv)

            time_end = time.time()
            print('{}  {}  {:.5f}'.format(idx, image_name, time_end - time_start))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
