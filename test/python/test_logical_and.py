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
"""nGraph TensorFlow bridge logical_and operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import tensorflow as tf


from common import NgraphTest


class TestLogicalAndOperations(NgraphTest):

  @pytest.mark.parametrize(("v1", "v2", "expected"),
                           ((True, True, [True]), (True, False, ([False],)), 
                           ([ False, True, False], [True], (np.array([[False,  True, False]]),))
))                            
  def test_logical_and(self, v1, v2, expected):
    print("TensorFlow version: ", tf.GIT_VERSION, tf.VERSION)

    val1 = tf.placeholder(tf.bool, shape=(None))
    val2 = tf.placeholder(tf.bool, shape=(None))

    with tf.device(self.test_device):
      out = tf.logical_and(val1, val2)

      with tf.Session(config=self.config) as sess:
        result = sess.run((out,), feed_dict={val1: (v1,), val2: (v2,)})
        print ("V1 ", v1, " V2 ", v2)
        print ("Result :", result)
        print ("Expected :", expected)
        assert np.allclose(result, expected)

