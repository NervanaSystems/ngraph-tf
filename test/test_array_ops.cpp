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
// Please ensure the alphabetical order while adding the test functions

// Test op: Fill
TEST(ArrayOps, Fill) {
  std::vector<std::vector<int>> input_sizes;  // 1-D or higher

  input_sizes.push_back({2, 3, 4});
  input_sizes.push_back({10, 10, 10});
  input_sizes.push_back({1, 5});
  input_sizes.push_back({0});
  input_sizes.push_back({2, 5, 1, 3, 1});

  vector<int> static_input_indexes = {0};  // has static input

  for (auto const& input_size : input_sizes) {
    Scope root = Scope::NewRootScope();

    int input_dim = input_size.size();
    Tensor shape(DT_INT32, TensorShape({input_dim}));
    AssignInputValues<int>(shape, input_size);

    // 0-D(scalar) value to fill the returned tensor
    Tensor input_data(DT_FLOAT, TensorShape({}));
    AssignInputValuesRandom(input_data);

    // Fill creates a tensor filled with scalar value
    // 1-D shape of the output tensor
    auto R = ops::Fill(root, shape, input_data);
    vector<DataType> output_datatypes = {DT_FLOAT};
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "Fill", static_input_indexes, output_datatypes,
                          sess_run_fetchoutputs);
    opexecuter.RunTest();
  }  // end of for loop
}  // end of op Fill

// Test op: ExpandDims, inserts a dimension of 1 into a tensor's shape
TEST(ArrayOps, ExpandDims) {
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  AssignInputValuesRandom(A);

  // axis at which the dimension will be inserted
  // should be -rank-1 <= axis <= rank
  vector<int> axis_ = {-1, 0};

  vector<int> static_input_indexes = {1};
  vector<DataType> output_datatypes = {DT_FLOAT};

  for (auto const& axis : axis_) {
    Scope root = Scope::NewRootScope();
    auto R = ops::ExpandDims(root, A, axis);
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "ExpandDims", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    opexecuter.RunTest();
  }

}  // end of test op ExpandDims

// Test op: PreventGradient
TEST(ArrayOps, PreventGradient) {
  Scope scope_cpu = Scope::NewRootScope();

  std::vector<std::vector<int64>> input_sizes;

  input_sizes.push_back({2, 3, 4, 20});
  input_sizes.push_back({10, 10, 10});
  input_sizes.push_back({1, 5});
  input_sizes.push_back({0});

  vector<int> static_input_indexes = {};

  for (auto const& input_size : input_sizes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(input_size));
    AssignInputValuesRandom<float>(input_data, -10.0, 20.0f);

    auto R = ops::PreventGradient(root, input_data);
    vector<DataType> output_datatypes = {DT_FLOAT};
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "PreventGradient", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op PreventGradient

// Test op: Shape, outputs the shape of a tensor
TEST(ArrayOps, Shape2D) {
  Scope root = Scope::NewRootScope();

  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 7.5f);

  vector<int> static_input_indexes = {};

  auto attrs = ops::Shape::Attrs();
  attrs.out_type_ = DT_INT64;
  auto R = ops::Shape(root, A, attrs);

  vector<DataType> output_datatypes = {DT_INT64};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Shape", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of op Shape2D

TEST(ArrayOps, Shape3D) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;
  int dim3 = 4;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2, dim3}));

  AssignInputValues(A, 7.5f);

  vector<int> static_input_indexes = {};
  auto R = ops::Shape(root, A);

  vector<DataType> output_datatypes = {DT_INT32};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "Shape", static_input_indexes, output_datatypes,
                        sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of op Shape3D

// Size op: returns the size of a tensor
// This test changes the default attribute of out_type_
TEST(ArrayOps, SizeOpAttrsChange) {
  std::vector<std::vector<int64>> input_shapes;
  input_shapes.push_back({0});
  input_shapes.push_back({2});
  input_shapes.push_back({6, 10});
  input_shapes.push_back({10, 10, 10});
  input_shapes.push_back({1, 7, 8, 10});
  input_shapes.push_back({2, 5, 1, 3, 1});

  vector<int> static_input_indexes = {};
  vector<DataType> output_datatypes = {DT_INT64};

  auto attrs = ops::Size::Attrs();
  attrs.out_type_ = DT_INT64;

  for (auto const& shapes : input_shapes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(shapes));
    AssignInputValuesRandom<float>(input_data, -10.0, 20.0f);

    auto R = ops::Size(root, input_data, attrs);
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "Size", static_input_indexes, output_datatypes,
                          sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op SizeAttrsChange

// Size op: returns the size of a tensor
TEST(ArrayOps, SizeOpDefault) {
  std::vector<std::vector<int64>> input_shapes;

  input_shapes.push_back({0});
  input_shapes.push_back({1});
  input_shapes.push_back({3, 8});
  input_shapes.push_back({10, 11, 12});
  input_shapes.push_back({1, 7, 8, 10});
  input_shapes.push_back({2, 5, 1, 3, 1});

  vector<int> static_input_indexes = {};
  // Size Op default output tyep is DT_INT32
  vector<DataType> output_datatypes = {DT_INT32};

  for (auto const& shape : input_shapes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(shape));
    AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);
    auto R = ops::Size(root, input_data);

    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Size", static_input_indexes, output_datatypes,
                          sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op SizeDefault

// Test slice op
TEST(ArrayOps, Slice) {
  std::vector<std::vector<int64>> input_shapes;
  input_shapes.push_back({1,2,4,1});

  std::vector<int64> begin ={0,0,2,0};
  std::vector<int64> size = {-1,-1,2,-1};

  vector<int> static_input_indexes = {1,2};
  vector<DataType> output_datatypes = {DT_FLOAT};

  for (auto const& shape : input_shapes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(shape));
    AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);
    Tensor begin_tensor(DT_INT64, TensorShape({4}));
    AssignInputValues(begin_tensor, begin);
    Tensor size_tensor(DT_INT64, TensorShape({4}));
    AssignInputValues(size_tensor,size);

    auto R = ops::Slice(root, input_data, begin_tensor,size_tensor);

    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "Slice", static_input_indexes, output_datatypes,
                          sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of op Slice

// SpaceToDepth op
TEST(ArrayOps, SpaceToDepthToOneElement) {
  std::map<std::vector<int64>, int> input_map;
  input_map.insert(pair<std::vector<int64>, int>({1,2,2,1}, 2));
  input_map.insert(pair<std::vector<int64>, int>({1,2,2,3}, 2));
  input_map.insert(pair<std::vector<int64>, int>({1,3,3,3}, 3));
  input_map.insert(pair<std::vector<int64>, int>({1,10,10,5}, 10));
  
  vector<int> static_input_indexes = {};
  vector<DataType> output_datatypes = {DT_FLOAT};

  map<std::vector<int64>, int> :: iterator iter;
  for(iter = input_map.begin(); iter != input_map.end(); iter++){
    std::vector<int64> shape = iter->first;
    int block_size = iter->second;

    Scope root = Scope::NewRootScope();
    Tensor input_data(DT_FLOAT, TensorShape(shape));
    AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);
    //PrintTensor(input_data);

    auto R = ops::SpaceToDepth(root, input_data, block_size);
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "SpaceToDepth", static_input_indexes, output_datatypes,
                          sess_run_fetchoutputs);
    // vector<Tensor> tf_outputs;
    // opexecuter.ExecuteOnTF(tf_outputs);
    // for(auto t: tf_outputs){
    //   PrintTensor(t);
    // }

    // vector<Tensor> ngraph_outputs;
    // opexecuter.ExecuteOnNGraph(ngraph_outputs);
    // for(auto t: ngraph_outputs){
    //   PrintTensor(t);
    // }
    opexecuter.RunTest();
  }  
}  // end of op SpaceToDepthToOneElementOp

// SpaceToDepth op
TEST(ArrayOps, SpaceToDepthToMultipleElementsOp) {
  std::map<std::vector<int64>, int> input_map;
  input_map.insert(pair<std::vector<int64>, int>({1,6,4,1}, 2));
  // input_map.insert(pair<std::vector<int64>, int>({1,2,2,3}, 2));
  // input_map.insert(pair<std::vector<int64>, int>({1,3,3,3}, 3));
  // input_map.insert(pair<std::vector<int64>, int>({1,10,10,5}, 10));
  
  vector<int> static_input_indexes = {};
  vector<DataType> output_datatypes = {DT_FLOAT};

  map<std::vector<int64>, int> :: iterator iter;
  for(iter = input_map.begin(); iter != input_map.end(); iter++){
    std::vector<int64> shape = iter->first;
    int block_size = iter->second;

    Scope root = Scope::NewRootScope();
    Tensor input_data(DT_FLOAT, TensorShape(shape));
    //AssignInputValuesRandom<float>(input_data, -10.0f, 10.0f);
    std::vector<float> l(24);
    std::iota(l.begin(), l.end(), 1.0);
    AssignInputValues<float>(input_data, l);
    cout << input_data.SummarizeValue(100) << endl;

    PrintTensor(input_data);

    auto R = ops::SpaceToDepth(root, input_data, block_size);
    std::vector<Output> sess_run_fetchoutputs = {R};
    OpExecuter opexecuter(root, "SpaceToDepth", static_input_indexes, output_datatypes,
                          sess_run_fetchoutputs);
    vector<Tensor> tf_outputs;
    opexecuter.ExecuteOnTF(tf_outputs);
    cout << "TF result " << endl;
    for(auto t: tf_outputs){
      //PrintTensor(t);
      cout << t.SummarizeValue(100) << endl;
    }

    vector<Tensor> ngraph_outputs;
    opexecuter.ExecuteOnNGraph(ngraph_outputs);
    cout << " Ngraph results " << endl;
    for(auto t: ngraph_outputs){
      cout << t.SummarizeValue(100) << endl;
      //PrintTensor(t);
    }
    opexecuter.RunTest();
  }  
}  // end of op SpaceToDepthToMultipleElementsOp

// Test op: Tile, constructs a tensor by tiling a given tensor
TEST(ArrayOps, Tile) {
  std::vector<std::vector<int64>> input_sizes;  // 1-D or higher

  input_sizes.push_back({2, 3, 4});
  input_sizes.push_back({10, 10, 10});
  input_sizes.push_back({1, 5});
  input_sizes.push_back({0});

  vector<int> static_input_indexes = {1};  // has static input

  for (auto const& input_size : input_sizes) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(input_size));
    AssignInputValuesRandom<float>(input_data, -5.0f, 10.0f);

    // Must be of type int32 or int64,
    // 1-D. Length must be the same as the number of dimensions in input
    int input_dim = input_size.size();
    Tensor multiples(DT_INT32, TensorShape({input_dim}));
    AssignInputValuesRandom<int32>(multiples, 0, 20);

    auto R = ops::Tile(root, input_data, multiples);
    vector<DataType> output_datatypes = {DT_FLOAT};
    std::vector<Output> sess_run_fetchoutputs = {R};

    OpExecuter opexecuter(root, "Tile", static_input_indexes, output_datatypes,
                          sess_run_fetchoutputs);

    opexecuter.RunTest();
  }
}  // end of test op Tile

// Unpacks the given dimension of a rank R tensor into a (R-1) tensor
TEST(ArrayOps, Unpack) {
  std::vector<std::vector<int64>> input_sizes;

  // rank > 0
  input_sizes.push_back({3, 2, 3});
  input_sizes.push_back({4, 3});
  input_sizes.push_back({3});

  std::vector<int64> axes({0, 1, 0});

  vector<int> static_input_indexes = {};
  for (auto i = 0; i < input_sizes.size(); ++i) {
    Scope root = Scope::NewRootScope();

    Tensor input_data(DT_FLOAT, TensorShape(input_sizes[i]));
    AssignInputValuesRandom<float>(input_data, -20, 50);

    ops::Unstack::Attrs attrs;
    attrs.axis_ = axes[i];

    auto R = ops::Unstack(root, input_data, input_sizes[i][axes[i]], attrs);

    // Unpack returns a list of Tensors
    // which internally flatten to multiple outputs
    // retrieve using indexes
    // the indexes matches the axies dimension of the input size
    std::vector<Output> sess_run_fetchoutputs = {R[0], R[1], R[2]};
    vector<DataType> output_datatypes = {DT_FLOAT, DT_FLOAT, DT_FLOAT};

    OpExecuter opexecuter(root, "Unpack", static_input_indexes,
                          output_datatypes, sess_run_fetchoutputs);

    opexecuter.RunTest();
  }  // end of for loop
}  // end of testing Unpack

// Test op: ZerosLike
// Returns a tensor of zeros of the same shape and type as the input tensor
TEST(ArrayOps, ZerosLike) {
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 3;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));

  AssignInputValues(A, 7.5f);

  vector<int> static_input_indexes = {};
  auto R = ops::ZerosLike(root, A);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  OpExecuter opexecuter(root, "ZerosLike", static_input_indexes,
                        output_datatypes, sess_run_fetchoutputs);

  opexecuter.RunTest();
}  // end of test op ZerosLike

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
