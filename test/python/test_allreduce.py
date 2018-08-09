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
"""nGraph TensorFlow bridge cast operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
import horovod.tensorflow as hvd
from common import NgraphTest
import numpy as np

hvd.init()
class TestAllreduceOp(NgraphTest):
  np_inp = np.random.rand(2,3)
  def test_allreduce(self):
    val = tf.placeholder(tf.float32, shape=(2,3))

    with self.device:
      out = hvd.allreduce(val)

      with self.session as sess:
        result = sess.run((out,), feed_dict={val: self.np_inp})
   
    with tf.device('/cpu:0'):
      out_cpu = hvd.allreduce(val)

      with self.session as sess:
        expected = sess.run((out_cpu,), feed_dict={val: self.np_inp})
    print("before allreduce", self.np_inp) 
    np.testing.assert_allclose(result, expected, rtol=5e-7)
    print(result)

