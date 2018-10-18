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

    def test_disable_assert(self):
        test_input = ((1, 1))
        x = tf.placeholder(tf.int32, shape=(2,))
        y = tf.placeholder(tf.int32, shape=(2,))
        z = tf.placeholder(tf.int32, shape=(2,))
        assert_op = tf.Assert(tf.less_equal(tf.reduce_max(z), 1), [x])

        with tf.control_dependencies([assert_op]):
            a2 = tf.add(x, y)

        def run_test(sess):
            return sess.run(
                a2, feed_dict={
                    x: test_input,
                    y: test_input,
                    z: test_input
                })

        assert (
            self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()
