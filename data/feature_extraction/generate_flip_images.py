# -*- coding: utf-8 -*-

import os
import glob
import time
import urllib2
import cv2
import tensorflow as tf


cwd = os.getcwd()
images_lists = sorted(glob.glob('mscoco/*.jpg'))

image_save_path = 'mscoco_flip/'

if os.path.isdir(os.path.join(cwd, image_save_path)) is False:
    os.mkdir(os.path.join(cwd, image_save_path))

sess = tf.InteractiveSession()

with sess.as_default():
    tf_image = tf.placeholder(tf.string, None)
    image_data = tf.image.decode_jpeg(tf_image, channels=3)

    tf_image_data_flipped = tf.image.flip_left_right(image_data)

    for idx, image_path in enumerate(images_lists):
            start_time = time.time()
            image_name = os.path.basename(image_path)

            current_image_save_path = os.path.join(image_save_path, image_name)
            url = 'file://' + os.path.join(cwd, image_path)
            image_string = urllib2.urlopen(url).read()

            encoded_image = sess.run(tf_image_data_flipped, feed_dict={tf_image: image_string})

            cv2.imwrite(current_image_save_path, cv2.cvtColor(encoded_image, cv2.COLOR_RGB2BGR))

            print("{}  {}  {:.5f}".format(idx+1, image_path, time.time()-start_time))
