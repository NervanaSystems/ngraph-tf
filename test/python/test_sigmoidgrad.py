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
"""nGraph TensorFlow bridge ReluGrad operation test

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops.gen_math_ops import sigmoid_grad

from common import NgraphTest

import numpy as np


class TestSigmoidGradOperations(NgraphTest):

    def test_sigmoidgrad_2d(self):
        y = constant_op.constant(
            self.generate_random_numbers(30, 1.0, 10.0), shape=[10,3])
        y_delta = constant_op.constant(
            self.generate_random_numbers(30, 0.0, 10.0), shape=[10,3])

        out = sigmoid_grad(y, y_delta)

        def run_test(sess):
            return sess.run(out)

        print("ngraph ", self.with_ngraph(run_test))
        print("cpu ", self.without_ngraph(run_test))

        assert np.allclose(
            self.with_ngraph(run_test), self.without_ngraph(run_test))


    def test_sigmoidgrad_3d(self):
        y = constant_op.constant(
            self.generate_random_numbers(450, 1.0, 15.0), shape=[10,3,15])
        y_delta = constant_op.constant(
            self.generate_random_numbers(450, -3.0, 10.0), shape=[10,3,15])

        out = sigmoid_grad(y, y_delta)

        def run_test(sess):
            return sess.run(out)

        print("ngraph ", self.with_ngraph(run_test))
        print("cpu ", self.without_ngraph(run_test))

        assert np.allclose(
            self.with_ngraph(run_test), self.without_ngraph(run_test))

    #@pytest.mark.parametrize("padding", ("VALID", "SAME"))
    # def test_nchw(self, padding):
    #     # The expected size of the backprop will depend on whether padding is VALID
    #     # or SAME.
    #     out_backprop_in_sizes = self.OUT_BACKPROP_IN_SIZES[padding]
    #     x1, x2 = self.make_filter_and_backprop_args(out_backprop_in_sizes)

    #     def run_test_ngraph(sess):
    #         t1 = constant_op.constant(self.INPUT_SIZES_NCHW)
    #         t2 = constant_op.constant(x2, shape=self.FILTER_IN_SIZES)
    #         t3 = constant_op.constant(x1, shape=out_backprop_in_sizes)
    #         inp = nn_ops.conv2d_backprop_input(
    #             t1,
    #             t2,
    #             t3,
    #             strides=[1, 1, 2, 2],
    #             padding=padding,
    #             data_format='NCHW')
    #         return sess.run(inp)

    #     # To validate on the CPU side we will need to run in NHWC, because the CPU
    #     # implementation of conv/conv backprop does not support NCHW. We will
    #     # transpose on the way in and on the way out.
    #     def run_test_tf(sess):
    #         t1 = constant_op.constant(self.INPUT_SIZES_NHWC)
    #         t2 = constant_op.constant(x2, shape=self.FILTER_IN_SIZES)
    #         t3 = constant_op.constant(x1, shape=out_backprop_in_sizes)
    #         t3 = tf.transpose(t3, [0, 2, 3, 1])
    #         inp = nn_ops.conv2d_backprop_input(
    #             t1,
    #             t2,
    #             t3,
    #             strides=[1, 2, 2, 1],
    #             padding=padding,
    #             data_format='NHWC')
    #         inp = tf.transpose(inp, [0, 3, 1, 2])
    #         return sess.run(inp)

    #     assert np.allclose(
    #         self.with_ngraph(run_test_ngraph), self.without_ngraph(run_test_tf))

    # def test_sigmoidgrad_2d(self):
    #     #gradients = constant_op.constant(
    #     #    self.generate_random_numbers(6, 1.0, 10.0), shape=[2, 3])
    #     y = constant_op.constant(
    #         self.generate_random_numbers(6, 1.0, 10.0), shape=[2, 3])
    #     y_delta = constant_op.constant(
    #         self.generate_random_numbers(6, 0.0, 100.0), shape=[2, 3])

    #     # Run on nGraph
    #     out = sigmoid_grad(y, y_delta)
    #     with self.session as sess:
    #     	result = sess.run(out)

    # Run on CPU


#ngraph.disable()
#if ngraph.is_enabled():
#	print ("ngraph is enabled not as expected")
#out = sigmoid_grad(y, y_delta)
#with self.session as sess:
#        expected = sess.run(out)

#assert (result == expected).all()

#def test_sigmoidgrad_1d(self):
#    gradients = constant_op.constant(
#        self.generate_random_numbers(100, 123.0, 345.0), shape=[100])
#    features = constant_op.constant(
#        self.generate_random_numbers(100, 567.0, 789.0), shape=[100])

#    # Run on nGraph
#    ngraph.disable()
#    out = sigmoid_grad(gradients, features)
#    with self.session as sess:
#    	result = sess.run(out)

#    # Run on CPU
#    #ngraph.disable()
#    #if ngraph.is_enabled():
#    #	print ("ngraph is enabled not as expected")
#    #out = sigmoid_grad(gradients, features)
#    #with self.session as sess:
#    #	expected = sess.run(out)

#    #assert (result == expected).all()
