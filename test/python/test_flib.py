# ==============================================================================
#  Copyright 2019 Intel Corporation
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
"""nGraph TensorFlow bridge function library test
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import tensorflow as tf
import numpy as np
from common import NgraphTest
from google.protobuf import text_format


def import_pbtxt(pb_filename):
    graph_def = tf.GraphDef()
    with open(pb_filename, "r") as f:
        text_format.Merge(f.read(), graph_def)

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph


def get_tensor(graph, tname):
    return graph.get_tensor_by_name("import/" + tname)


class TestFlibOperations(NgraphTest):

    def test_flib_1(self):
        graph = import_pbtxt('flib_graph_1.pbtxt')
        with graph.as_default() as g:

            x = get_tensor(g, "Placeholder:0")
            y = get_tensor(g, "Placeholder_1:0")
            z = get_tensor(g, "Placeholder_2:0")

            a = get_tensor(g, "add_1:0")
            b = get_tensor(g, "Sigmoid:0")

            sess_fn = lambda sess: sess.run(
                [a, b], feed_dict={i: np.full((2, 3), 1.0) for i in [x, y, z]})

            res1 = self.with_ngraph(sess_fn)
            res2 = self.without_ngraph(sess_fn)
            exp = [np.full((2, 3), 3.0), np.full((2, 3), 0.95257413)]
            # Note both run on Host (because NgraphEncapsulate can only run on host)
            assert np.isclose(res1, res2).all()
            # Comparing with expected value
            assert np.isclose(res1, exp).all()

    @pytest.mark.skip(reason="Not passing through grappler")
    def test_flib_2(self):
        graph = import_pbtxt('flib_graph_2.pbtxt')
        with graph.as_default() as g:

            x = get_tensor(g, "Variable_2/peek/_2:0")
            y = get_tensor(g, "Variable_1/peek/_3:0")
            z = get_tensor(g, "Variable/peek/_4:0")

            a = get_tensor(g, "add_1:0")
            b = get_tensor(g, "Sigmoid:0")

            def sess_fn(sess):
                #sess.run(tf.global_variables_initializer())
                return sess.run(
                    [a, b],
                    feed_dict={i: np.full((2, 3), 1.0) for i in [x, y, z]})

            res1 = self.with_ngraph(sess_fn)
            res2 = self.without_ngraph(sess_fn)
            exp = [np.full((2, 3), 3.0), np.full((2, 3), 0.95257413)]
            # Note both run on Host (because NgraphEncapsulate can only run on host)
            assert np.isclose(res1, res2).all()
            # Comparing with expected value
            assert np.isclose(res1, exp).all()  #fails
