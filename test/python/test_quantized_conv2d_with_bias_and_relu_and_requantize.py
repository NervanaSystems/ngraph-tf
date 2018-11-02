from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np

import tensorflow as tf
from common import NgraphTest

from tensorflow.python.ops import nn_ops


# The op does not have a C++ api, hence wirting a ython test
# Since it is not yet in the master TF tree, keeping it disabled
class TestQuantizedConv2DWithBiasAndReluAndRequantize(NgraphTest):
    # TODO: parameterize this function
    def test_depthwise_conv2d(self):
        dim1 = 2
        dim2 = 3
        in_channels = 2
        out_channels = 1
        filt_h = 2
        filt_w = 2
        batchsize = 1

        inshape = (batchsize, dim1, dim2, in_channels)
        inp = tf.placeholder(tf.quint8, shape=inshape)
        filtshape = (filt_h, filt_w, in_channels, out_channels)
        filter = tf.placeholder(tf.qint8, shape=filtshape)
        biasshape = (out_channels,)
        bias = tf.placeholder(tf.float32, shape=(out_channels,))

        in_array = np.arange(np.prod(inshape)).astype('uint8').reshape(inshape)
        filt_array = np.arange(
            -5, -5 + np.prod(filtshape)).astype('int8').reshape(filtshape)
        bias_array = np.arange(
            np.prod(biasshape)).astype('float32').reshape(biasshape)

        min_input = tf.constant(0, dtype=tf.float32)
        max_input = tf.constant(np.prod(inshape), dtype=tf.float32)
        min_filter = tf.constant(0, dtype=tf.float32)
        max_filter = tf.constant(np.prod(filtshape), dtype=tf.float32)
        min_freezed_output = tf.constant(-5000, dtype=tf.float32)
        max_freezed_output = tf.constant(5000, dtype=tf.float32)
        strides = [1, 1, 1, 1]
        padding = "SAME"
        out_type = tf.quint8
        dilations = [1, 1, 1, 1]

        conv = nn_ops.quantized_conv2d_with_bias_and_relu_and_requantize(
            inp, filter, bias, min_input, max_input, min_filter, max_filter,
            min_freezed_output, max_freezed_output, strides, padding, out_type,
            dilations)
        sess_fn = lambda sess: sess.run(conv, feed_dict={inp: in_array, filter: filt_array, bias: bias_array})

        #assert np.isclose(
        #    self.with_ngraph(sess_fn), self.without_ngraph(sess_fn)).all()
        print(self.without_ngraph(sess_fn))
