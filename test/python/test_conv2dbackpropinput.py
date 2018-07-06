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
"""nGraph TensorFlow bridge depthwise_conv2d operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops

from common import NgraphTest


class TestConv2DBackpropInput(NgraphTest):
  @pytest.mark.parametrize("padding", ("VALID", "SAME"))
  def test_input(self, padding):
    input_sizes = [1, 2, 4, 3]
    filter_in_sizes = [2, 2, 2, 2]
    out_backprop_in_sizes = [1, 2, 3, 2]
    total_size_1 = 1
    total_size_2 = 1

    for s in out_backprop_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s

    x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
    x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]

    import pdb; pdb.set_trace()

    with self.device:
      with tf.Session(config=self.config) as sess:
        t1 = constant_op.constant(input_sizes)
        t2 = constant_op.constant(x2, shape=filter_in_sizes)
        t3 = constant_op.constant(x1, shape=out_backprop_in_sizes)
        inp = nn_ops.conv2d_backprop_input(
            t1, t2, t3, strides=[1, 1, 1, 1], padding=padding)
        value = sess.run(inp)

    with self.session as sess:
      t1 = constant_op.constant(input_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes)
      t2 = constant_op.constant(x1, shape=out_backprop_in_sizes)
      inp = nn_ops.conv2d_backprop_input(
          t1, t2, t3, strides=[1, 1, 1, 1], padding=padding)
      expected = sess.run(inp)

    assert (value == expected).all()
