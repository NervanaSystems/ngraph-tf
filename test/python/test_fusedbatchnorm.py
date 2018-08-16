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
"""nGraph TensorFlow FusedBatchNorm test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from common import NgraphTest


class TestFusedBatchNorm(NgraphTest):
  x = np.random.rand(64, 3, 10, 8).astype('f')
  scale = [1.0, 0.9, 1.1]
  offset = [0.1, 0.2, -.3]

  def test_fusedbatchnorm_nchw(self):
    with self.device:
      norm = tf.nn.fused_batch_norm(self.x, self.scale, self.offset,
                                    data_format='NCHW')

      with self.session as sess:
        result = sess.run(norm)

    with tf.device('/cpu:0'):
      x_t = tf.transpose(self.x, (0, 2, 3, 1))
      # tensorflow CPU doesn't support NCHW
      norm = tf.nn.fused_batch_norm(x_t, self.scale, self.offset,
                                    data_format='NHWC')

      with self.session as sess:
        expected = sess.run(norm)

    np.testing.assert_allclose(result[0],
                               np.transpose(expected[0], (0, 3, 1, 2)),
                               rtol=0.01)
    np.testing.assert_allclose(result[1], expected[1], rtol=0.01)
    np.testing.assert_allclose(result[2], expected[2], rtol=0.01)

  def test_fusedbatchnorm_nhwc(self):
    x_t = tf.transpose(self.x, (0, 2, 3, 1))

    with self.device:
      norm = tf.nn.fused_batch_norm(x_t, self.scale, self.offset,
                                    data_format='NHWC')

      with self.session as sess:
        result = sess.run(norm)

    with tf.device('/cpu:0'):
      norm = tf.nn.fused_batch_norm(x_t, self.scale, self.offset,
                                    data_format='NHWC')

      with self.session as sess:
        expected = sess.run(norm)

    np.testing.assert_allclose(result[0], expected[0], rtol=0.01)
    np.testing.assert_allclose(result[1], expected[1], rtol=0.01)
    np.testing.assert_allclose(result[2], expected[2], rtol=0.01)
