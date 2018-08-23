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

#include "TestCaseBuilderSimple.h"
#include "TestUtilities.h"
#include "gtest/gtest.h"

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

namespace testing{

#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());

TEST(TestSimple, SimpleDEAdd) {
  // Create a tf graph
  Scope root = Scope::NewRootScope();
  int dim1 = 2;
  int dim2 = 2;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));
  //auto C = ops::Mul(root,A, B);
  DummyAssignInputValues(A, 2.1f);
  DummyAssignInputValues(B, 4.1f);

  auto R = ops::Add(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT};
  std::vector<Output> sess_run_fetchoutputs = {R};
  BuilderTestSimple buildertest(root, "Add", output_datatypes, sess_run_fetchoutputs);

  buildertest.ExecuteOnNGraph();
  buildertest.ExecuteOnTF();
  buildertest.CompareNgraphAndTF();
}

TEST(TestSimple, SimpleDESparseSoftmax) {
  Scope root = Scope::NewRootScope();
  int batch = 1000;
  int num_of_classes = 200;

  Tensor A(DT_FLOAT, TensorShape({batch, num_of_classes}));
  Tensor B(DT_INT32, TensorShape({batch}));

  DummyAssignInputValues(A, 2.0f);
  DummyAssignInputIntValues(B, num_of_classes);

  auto R = ops::SparseSoftmaxCrossEntropyWithLogits(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT, DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R.loss, R.backprop};
  BuilderTestSimple buildertest(root, "SparseSoftmaxCrossEntropyWithLogits", output_datatypes, sess_run_fetchoutputs);

  buildertest.ExecuteOnNGraph();
  buildertest.ExecuteOnTF();
  buildertest.CompareNgraphAndTF();
}

TEST(TestSimple, SimpleDERealDiv) {
  Scope root = Scope::NewRootScope();
  int dim1 = 100;
  int dim2 = 200;

  Tensor A(DT_FLOAT, TensorShape({dim1, dim2}));
  Tensor B(DT_FLOAT, TensorShape({dim1, dim2}));

  DummyAssignInputValues(A, 2.0f);
  DummyAssignInputValues(B, 7.0f);
  
  auto R = ops::RealDiv(root, A, B);

  vector<DataType> output_datatypes = {DT_FLOAT};

  std::vector<Output> sess_run_fetchoutputs = {R};
  BuilderTestSimple buildertest(root, "RealDiv", output_datatypes, sess_run_fetchoutputs);

  buildertest.ExecuteOnNGraph();
  buildertest.ExecuteOnTF();
  buildertest.CompareNgraphAndTF();
}

} // namespace testing

}  // namespace ngraph_bridge
}  // namespace tensorflow
