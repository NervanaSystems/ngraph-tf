/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

#include "tf_graph_writer.h"
#include "test_utilities.h"
#include "gtest/gtest.h"
#include "ngraph_assign_clusters.h"
#include "ngraph_backend_manager.h"
#include "ngraph_mark_for_clustering.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());
#define ASSERT_NOT_OK(x) ASSERT_NE((x), ::tensorflow::Status::OK());

// Simple Graph
TEST(Variables, SmallGraph1) {
  Scope root = Scope::NewRootScope();

  PartialTensorShape varShape({2, 2});
  auto var = ops::Variable(root, varShape, DT_FLOAT);
  auto init_value = ops::Const(root, {{1.f, 1.f}, {1.f, 1.f}});
  auto var_assign = ops::Assign(root, var, init_value);

  auto c = ops::Const(root, {{1.f, 1.f}, {1.f, 1.f}});

  auto add = ops::Add(root, var, c);

  auto assign = ops::Assign(root, var, add);

  // Turn off optimizations so that all the nodes are processed
  tensorflow::SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(tensorflow::OptimizerOptions_Level_L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(tensorflow::RewriterConfig::OFF);

  // Run on nGraph
  ActivateNGraph();
  ClientSession ng_session(root, options);
  std::vector<tensorflow::Tensor> ng_outputs1;
  std::vector<tensorflow::Tensor> ng_outputs2;
  std::vector<tensorflow::Tensor> ng_outputs3;

  ASSERT_OK(ng_session.Run(
      {
          var_assign,
      },
      &ng_outputs1));
  for (int i = 0; i < 2; i++) {
    ASSERT_OK(ng_session.Run({assign}, &ng_outputs2));
  }

  ASSERT_OK(ng_session.Run({var}, &ng_outputs3));
  
  // Run on TF
  DeactivateNGraph();
  ClientSession tf_session(root, options);
  std::vector<tensorflow::Tensor> tf_outputs1;
  std::vector<tensorflow::Tensor> tf_outputs2;
  std::vector<tensorflow::Tensor> tf_outputs3;

  ASSERT_OK(tf_session.Run(
      {
          var_assign,
      },
      &tf_outputs1));
  for (int i = 0; i < 2; i++) {
    ASSERT_OK(tf_session.Run({assign}, &tf_outputs2));
  }

  ASSERT_OK(tf_session.Run({var}, &tf_outputs3));
  
  Compare(tf_outputs1, ng_outputs1);
  Compare(tf_outputs2, ng_outputs2);
  Compare(tf_outputs3, ng_outputs3);

  //For other test cases  
  ActivateNGraph();
}



}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow