from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import numpy as np
import itertools
import glob

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import common
import ngraph_bridge
from tensorflow.python.framework import constant_op

FLAGS = None


def save_dummy():
    W = np.full(shape=[5, 5, 1, 5], fill_value=1, dtype=np.float32)
    np.save('dummy.npy', W)


def const_test():
    x = tf.placeholder(tf.float32, [2])
    y = tf.constant(1.2, dtype=tf.float32, shape=[2])
    #y = constant_op.constant([3, 4])

    with tf.Session() as sess:
        W = np.full(shape=[2], fill_value=1, dtype=np.float32)
        y_eval = y.eval(feed_dict={x: W})

    print('ok')


def main():

    #const_test()
    #return

    save_dummy()

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First conv layer: maps one grayscale image to 5 feature maps of 13 x 13
    W_conv1 = tf.constant(
        1.2, dtype=tf.float32, shape=[5, 5, 1, 5], name='fabi_const')
    conv = common.conv2d_stride_2_valid(x_image, W_conv1)

    with tf.Session() as sess:
        start_time = time.time()
        x_test = mnist.test.images[:FLAGS.batch_size]
        # Run model
        y_conv_val = conv.eval(feed_dict={x: x_test})
        elasped_time = time.time() - start_time
        print("total time(s)", elasped_time)

    print("ok")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/mnist/input_data',
        help='Directory where input data is stored')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument(
        '--test_image_count',
        type=int,
        default=None,
        help="Number of test images to evaluate on")
    parser.add_argument(
        '--save_batch',
        type=bool,
        default=False,
        help='Whether or not to save the test image and label.')
    parser.add_argument(
        '--report_accuracy',
        type=bool,
        default=False,
        help='Whether or not to save the compute the test accuracy.')

    FLAGS, unparsed = parser.parse_known_args()
    main()