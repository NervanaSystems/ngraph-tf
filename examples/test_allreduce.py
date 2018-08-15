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
import numpy as np
print("before init")
hvd.init()

val = tf.placeholder(tf.float32, shape=(1,))

with tf.device('/device:NGRAPH:0'):
  out = hvd.allreduce(val)

with tf.Session() as sess:
  result = sess.run((out,), feed_dict={val: (5.5,)})
   
with tf.device('/cpu:0'):
  out_cpu = hvd.allreduce(val)

  with tf.Session() as sess:
    expected = sess.run((out_cpu,), feed_dict={val: (5.5,)})
    
np.testing.assert_allclose(result, expected, rtol=5e-7)
print("Finishes")

