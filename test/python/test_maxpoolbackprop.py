# ==============================================================================
#  Copyright 2018 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
"""nGraph TensorFlow bridge MaxPoolBackprop operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np

import tensorflow as tf
from tensorflow.python.ops.gen_nn_ops import max_pool_grad

from common import NgraphTest


class TestMaxPoolBackpropInput(NgraphTest):
  strides = [1, 2, 3, 1]
  input_nhwc = np.random.rand(128, 224, 224, 3)
  input_nchw = np.random.rand(128, 3, 224, 224)
  grad_nhwc = {"VALID": np.random.rand(
      128, 112, 74, 3), "SAME": np.random.rand(128, 112, 75, 3)}
  grad_nchw = {"VALID": np.random.rand(
      128, 3, 74, 224), "SAME": np.random.rand(128, 3, 75, 224)}

  @pytest.mark.parametrize("padding", ("VALID", "SAME"))
  def test_nhwc(self, padding):
    ksize = [1, 2, 3, 1]
    output = np.random.rand(*ksize)
    np_nhwc = self.grad_nhwc[padding]
    if padding == "VALID":
      grad = tf.placeholder(tf.float32, shape=(128, 112, 74, 3))
    elif padding == "SAME":
      grad = tf.placeholder(tf.float32, shape=(128, 112, 75, 3))

    with self.device:
      a = max_pool_grad(self.input_nhwc, output, grad, ksize, self.strides,
                        padding=padding, data_format="NHWC")
      with self.session as sess:
        result = sess.run(a, feed_dict={grad: np_nhwc})

    with tf.device('/cpu:0'):
      b = max_pool_grad(self.input_nhwc, output, grad, ksize, self.strides,
                        padding=padding, data_format="NHWC")
      with self.session as sess:
        (expected) = sess.run(b, feed_dict={grad: np_nhwc})

    np.testing.assert_allclose(result, expected, rtol=5e-7)

  @pytest.mark.parametrize("padding", ("VALID", "SAME"))
  def test_nchw(self, padding):
    ksize = [1, 1, 3, 1]
    output = np.random.rand(*ksize)
    np_nchw = self.grad_nchw[padding]
    if padding == "VALID":
      grad = tf.placeholder(tf.float32, shape=(128, 3, 74, 224))
    elif padding == "SAME":
      grad = tf.placeholder(tf.float32, shape=(128, 3, 75, 224))

    with self.device:
      a = max_pool_grad(self.input_nchw, output, grad, ksize, self.strides,
                        padding=padding, data_format="NCHW")
      with self.session as sess:
        (result) = sess.run(a, feed_dict={grad: np_nchw})
    # To validate on the CPU side we will need to run in NHWC, because the CPU
    # implementation of avgpool backprop does not support NCHW. We will
    # transpose on the way in and on the way out
    with tf.device('/cpu:0'):
      grad = tf.transpose(grad, [0, 2, 3, 1])
      np_nchw = np.transpose(np_nchw, [0, 2, 3, 1])
      ksize = [1, 3, 1, 1]
      output = np.random.rand(*ksize)
      strides = [1, 3, 1, 2]
      b = max_pool_grad(self.input_nchw, output, grad, ksize, strides,
                        padding=padding, data_format="NHWC")
      b = tf.transpose(b, [0, 3, 1, 2])
      with self.session as sess:
        expected = sess.run(b, feed_dict={grad: np_nchw})

    np.testing.assert_allclose(result, expected, rtol=5e-7)
