/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "gtest/gtest.h"
#include "opexecuter.h"
#include "test_utilities.h"

#include "ngraph_utils.h"
#include "tf_graph_writer.h"

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());

// Test(TestCaseName, TestName)
// Please ensure
// Neither TestCaseName nor TestName should contain underscore
// https://github.com/google/googletest/blob/master/googletest/docs/primer.md
// Use only Tensors and ops::Const() to provide input to the test op

TEST(NNOps, SparseSoftmaxCrossEntropyWithLogits) {
  Scope root = Scope::NewRootScope();
  int batch = 1000;
  int num_of_classes = 200;

  Tensor A(DT_FLOAT, TensorShape({batch, num_of_classes}));
  Tensor B(DT_INT32, TensorShape({batch}));

  AssignInputValues(A, 2.0f);
  AssignInputIntValues(B, num_of_classes);

  vector<int> static_input_indexes = {};
  auto R = ops::SparseSoftmaxCrossEntropyWithLogits(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT, DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R.loss, R.backprop};
  OpExecuter opexecuter(root, "SparseSoftmaxCrossEntropyWithLogits",
                        static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.ExecuteOnNGraph();
  opexecuter.ExecuteOnTF();
  opexecuter.CompareNGraphAndTF();
}

TEST(NNOps, Conv2DBackpropFilter) {
  Scope root = Scope::NewRootScope();

  // TF Default formats
  // Input NHWC :[batch, in_height, in_width, in_channels]
  std::vector<int64> input_size_NHWC = {1, 7, 6, 2};
  // Filter :[filter_height, filter_width, in_channels, out_channels]
  std::vector<int64> filter_size_HWIO = {3, 3, 2, 2};
  // Out_delta :[batch, out_height, out_width, out_channels]
  std::vector<int64> output_del_size_valid = {1, 3, 2, 2};
  std::vector<int64> output_del_size_same = {1, 4, 3, 2};
  Tensor output_delta_valid(DT_FLOAT, TensorShape(output_del_size_valid));
  Tensor output_delta_same(DT_FLOAT, TensorShape(output_del_size_same));
  AssignInputValues(output_delta_valid, -1.1f);
  AssignInputValues(output_delta_same, -1.1f);

  std::map<std::string, Tensor*> out_delta_size_map = {
      {"VALID", &output_delta_valid}, {"SAME", &output_delta_same}};

  std::vector<int> stride = {1, 2, 2, 1};
  Tensor input_data(DT_FLOAT, TensorShape(input_size_NHWC));
  AssignInputValues(input_data, -1.1f);

  auto filter_sizes = ops::Const(root, {3, 3, 2, 2});

  ClientSession session(root);

  vector<int> static_input_indexes = {1};
  // TEST NHWC : default data format
  for (auto map_iterator : out_delta_size_map) {
    auto padding_type = map_iterator.first;
    auto output_delta = *(out_delta_size_map[padding_type]);

    auto R = ops::Conv2DBackpropFilter(root, input_data, filter_sizes,
                                       output_delta, stride, padding_type);

    vector<DataType> output_datatypes = {DT_FLOAT};
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Conv2DBackpropFilter", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    opexecuter.ExecuteOnNGraph();
    opexecuter.ExecuteOnTF();
    opexecuter.CompareNGraphAndTF();
    break;
  }

  /*
  // TEST NCHW
  // Dialtion rates > 1 not supported on CPU
  // Current testing only with dialtion rate 1
  ops::Conv2DBackpropFilter::Attrs op_attr_nchw;
  op_attr_nchw = op_attr_nchw.DataFormat("NCHW");
  op_attr_nchw = op_attr_nchw.Dilations({1, 1, 1, 1});

  ops::Conv2DBackpropFilter::Attrs op_attr_nhwc;
  op_attr_nhwc = op_attr_nhwc.DataFormat("NHWC");
  op_attr_nhwc = op_attr_nhwc.Dilations({1, 1, 1, 1});

  for (auto map_iterator : out_delta_size_map) {
    auto padding_type = map_iterator.first;
    auto output_delta = *(out_delta_size_map[padding_type]);

    auto input_data_NCHW = ops::Transpose(root, input_data, {0, 3, 1, 2});
    auto output_delta_NCHW = ops::Transpose(root, output_delta, {0, 3, 1, 2});
    auto stride_NCHW(stride);
    stride_NCHW[1] = stride[3];
    stride_NCHW[2] = stride[1];
    stride_NCHW[3] = stride[2];

    auto r_ngraph = ops::Conv2DBackpropFilter(
        root_ngraph.WithOpName("r_NGRAPH"), input_data_NCHW, filter_sizes,
        output_delta_NCHW, stride_NCHW, padding_type, op_attr_nchw);

    // CPU supports only NHWC
    auto r_cpu = ops::Conv2DBackpropFilter(root.WithOpName("r_CPU"), input_data,
                                           filter_sizes, output_delta, stride,
                                           padding_type, op_attr_nhwc);

    ASSERT_OK(session.Run({r_ngraph}, &outputs_ngraph));
    ASSERT_OK(session.Run({r_cpu}, &outputs_cpu));

    ASSERT_EQ(outputs_ngraph[0].shape(), outputs_cpu[0].shape());
    AssertTensorEquals(outputs_ngraph[0], outputs_cpu[0]);
  }
  */
}

}  // namespace testing

}  // namespace ngraph_bridge
}  // namespace tensorflow
