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
"""nGraph TensorFlow bridge abs operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
import os

from common import NgraphTest


class TestAssertOperations(NgraphTest):

    def test_assert(self):
        x = tf.constant([1,2])
        y = tf.constant([1,2])
        z = tf.constant([1,1])
        assert_op = tf.Assert(tf.less_equal(tf.reduce_max(z), 1), [z])

        with tf.control_dependencies([assert_op]):
            a2 = tf.add(z, y)
            #a1 = tf.add(x, y)

        #a3 = tf.add(y,y)   
        def run_test(sess):
            return sess.run(a2)

        self.with_ngraph(run_test)

