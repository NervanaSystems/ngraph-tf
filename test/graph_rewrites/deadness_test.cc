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
#include "../test_utilities.h"
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

namespace testing {

#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());

// Replicate the deadness test mentioned here
// https://github.com/tensorflow/tensorflow/commit/6619dd5fdcad02f087f5758083e2585bdfef9e78#diff-77b7f1b6308c3eed108508e1a5d8f8dc

TEST(DeadnessCheck, livedead1NGRAPH) {
  Scope root = Scope::NewRootScope();

  auto A = ops::Placeholder(root, DataType::DT_FLOAT);
  auto B = ops::Placeholder(root, DataType::DT_FLOAT);
  auto C = ops::Placeholder(root, DataType::DT_FLOAT);
  auto pred = ops::Placeholder(root, DataType::DT_BOOL);

  auto S = ops::Switch(root, A, pred);
  auto P = ops::Add(root, A, B);

  auto Q = ops::Add(root, A, C);
  auto R = ops::Sub(root, S.output_true, B);

  auto M = ops::Mul(root, P, Q);
  auto D = ops::RealDiv(root, Q, R);

  std::vector<Tensor> outputs;
  ClientSession session(root);

  ASSERT_OK(session.Run(
      {{A, {3.f, 5.f}}, {B, {3.f, 2.f}}, {C, {3.f, 2.f}}, {pred, false}},
      {M, D}, &outputs));

  LOG(INFO) << outputs[0].flat<float>();
  LOG(INFO) << outputs[1].flat<float>();
}

TEST(DeadnessCheck, livedead1TF) {
  Scope root = Scope::NewRootScope();
  DeactivateNGraph();

  auto A = ops::Placeholder(root, DataType::DT_FLOAT);
  auto B = ops::Placeholder(root, DataType::DT_FLOAT);
  auto C = ops::Placeholder(root, DataType::DT_FLOAT);
  auto pred = ops::Placeholder(root, DataType::DT_BOOL);

  auto S = ops::Switch(root, A, pred);
  auto P = ops::Add(root, A, B);

  auto Q = ops::Add(root, A, C);
  auto R = ops::Sub(root, S.output_true, B);

  auto M = ops::Mul(root, P, Q);
  auto D = ops::RealDiv(root, Q, R);

  std::vector<Tensor> outputs;
  ClientSession session(root);

  ASSERT_OK(session.Run(
      {{A, {3.f, 5.f}}, {B, {3.f, 2.f}}, {C, {3.f, 2.f}}, {pred, false}},
      {M, D}, &outputs));

  LOG(INFO) << outputs[0].flat<float>();
  LOG(INFO) << outputs[1].flat<float>();
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
