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
"""nGraph TensorFlow bridge gather operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
import os

from common import NgraphTest

class TestGatherOperations(NgraphTest):

    def test_gather(self):
        val = tf.placeholder(tf.float32, shape=(5,))
        out = tf.gather(val, [2,1])

        def run_test(sess):
            return sess.run((out,), feed_dict={val: (10.0, 20.0, 30.0, 40.0, 50.0)})[0]


        assert (self.with_ngraph(run_test) == self.without_ngraph(run_test)).all()