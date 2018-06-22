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
"""nGraph TensorFlow bridge less operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import tensorflow as tf


from common import NgraphTest


class TestLessOperations(NgraphTest):

  @pytest.mark.parametrize(("v1", "v2", "expected"),
                           ((1.4, 1.0, [False]), (-1.0, -1.0, ([False],)), 
                           (-1.0, 1000, [True] ), (200, 200, ([False],)),
                           ([ -1.0, 1.0, -4], [0.1, 0.1, -4], (np.array([[True,  False, False]]),)),
                           ([ -1.0, 1.0, -4], [-1.0], (np.array([[False,  False, True]]),))
))                            
  def test_less(self, v1, v2, expected):
    print("TensorFlow version: ", tf.GIT_VERSION, tf.VERSION)

    val1 = tf.placeholder(tf.float32, shape=(None))
    val2 = tf.placeholder(tf.float32, shape=(None))

    with tf.device(self.test_device):
      out = tf.less(val1, val2)

      with tf.Session(config=self.config) as sess:
        result = sess.run((out,), feed_dict={val1: (v1,), val2: (v2,)})
        print ("V1 ", v1, " V2 ", v2)
        print ("Result :", result)
        print ("Expected :", expected)
        assert np.allclose(result, expected)

