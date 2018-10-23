# -*- coding: utf-8 -*-

import os
import glob
import time
import urllib2
import cv2
import numpy as np
import tensorflow as tf
from multiprocessing import Process


def worker(crop_format, images_lists, image_save_path):
    pid = os.getpid()
    if os.path.isdir(image_save_path) is False:
        os.mkdir(image_save_path)

    if crop_format == 'top_right':
        boxes = tf.convert_to_tensor([[0.0, 0.1, 0.9, 1.0]], dtype=np.float32)
    elif crop_format == 'top_left':
        boxes = tf.convert_to_tensor([[0.0, 0.0, 0.9, 0.9]], dtype=np.float32)
    elif crop_format == 'bottom_right':
        boxes = tf.convert_to_tensor([[0.1, 0.1, 1.0, 1.0]], dtype=np.float32)
    elif crop_format == 'bottom_left':
        boxes = tf.convert_to_tensor([[0.1, 0.0, 1.0, 0.9]], dtype=np.float32)
    else:
        raise Exception("crop_format supported: {}".format(crop_format))

    box_ind = tf.convert_to_tensor([0], dtype=tf.int32)
    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    with sess.as_default():
        tf_image = tf.placeholder(tf.string, None)
        tf_size = tf.placeholder(tf.int32, [2])

        image_data = tf.image.decode_jpeg(tf_image, channels=3)
        image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32) * 255.0
        image_data = tf.expand_dims(image_data, axis=0)

        tf_image_data = tf.image.crop_and_resize(image=image_data, boxes=boxes, box_ind=box_ind, crop_size=tf_size)

        for idx, image_path in enumerate(images_lists):
            start_time = time.time()

            image_name = os.path.basename(image_path)
            current_image_save_path = os.path.join(image_save_path, image_name)

            url = 'file://' + image_path
            image_string = urllib2.urlopen(url).read()
            # image_string = tf.gfile.FastGFile(image_path, 'r').read()

            im = cv2.imread(image_path, cv2.IMREAD_COLOR)
            im_shape = im.shape

            feed_dict = {tf_image: image_string, tf_size: [im_shape[0], im_shape[1]]}

            encoded_image = sess.run(tf_image_data, feed_dict)
            encoded_image = np.squeeze(encoded_image)
            cv2.imwrite(current_image_save_path, cv2.cvtColor(encoded_image, cv2.COLOR_RGB2BGR))

            print("{} {}  {}  {:.5f}".format(pid, idx, current_image_save_path, time.time()-start_time))


if __name__ == '__main__':

    p=[]
    images_lists = sorted(glob.glob('/data1/ailab_view/wenhaojiang/data/mscoco_flip/*.jpg'))
    save_path = '/data1/ailab_view/wenhaojiang/data/mscoco_flip_crop_top_right'
    # worker('top_right', images_lists, save_path)
    p.append(Process(target=worker, args=('top_right', images_lists, save_path)))

    save_path = '/data1/ailab_view/wenhaojiang/data/mscoco_flip_crop_top_left'
    # worker('top_left', images_lists, save_path)
    p.append(Process(target=worker, args=('top_left', images_lists, save_path)))

    save_path = '/data1/ailab_view/wenhaojiang/data/mscoco_flip_crop_bottom_right'
    # worker('bottom_right', images_lists, save_path)
    p.append(Process(target=worker, args=('bottom_right', images_lists, save_path)))

    save_path = '/data1/ailab_view/wenhaojiang/data/mscoco_flip_crop_bottom_left'
    # worker('bottom_left', images_lists, save_path)
    p.append(Process(target=worker, args=('bottom_left', images_lists, save_path)))


    images_lists = sorted(glob.glob('/data1/ailab_view/wenhaojiang/data/mscoco/*.jpg'))
    save_path = '/data1/ailab_view/wenhaojiang/data/mscoco_crop_top_right'
    # worker('top_right', images_lists, save_path)
    p.append(Process(target=worker, args=('top_right', images_lists, save_path)))

    save_path = '/data1/ailab_view/wenhaojiang/data/mscoco_crop_top_left'
    # worker('top_left', images_lists, save_path)
    p.append(Process(target=worker, args=('top_left', images_lists, save_path)))

    save_path = '/data1/ailab_view/wenhaojiang/data/mscoco_crop_bottom_right'
    # worker('bottom_right', images_lists, save_path)
    p.append(Process(target=worker, args=('bottom_right', images_lists, save_path)))

    save_path = '/data1/ailab_view/wenhaojiang/data/mscoco_crop_bottom_left'
    # worker('bottom_left', images_lists, save_path)
    p.append(Process(target=worker, args=('bottom_left', images_lists, save_path)))

    for ele in p:
        ele.start()

    for ele in p:
        ele.join()
