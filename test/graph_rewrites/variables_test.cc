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
#include "ngraph_assign_clusters.h"
#include "ngraph_backend_manager.h"
#include "ngraph_mark_for_clustering.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tf_graph_writer.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());
#define ASSERT_NOT_OK(x) ASSERT_NE((x), ::tensorflow::Status::OK());

TEST(Variables, SmallGraph1) {
  Scope root = Scope::NewRootScope();

  PartialTensorShape varShape({2, 2});
  auto var = ops::Variable(root.WithOpName("Var1"), varShape, DT_FLOAT);
  auto init_value = ops::Const(root, {{1.f, 1.f}, {1.f, 1.f}});
  auto var_assign = ops::Assign(root, var, init_value);

  //   TensorShape constShape({2,2});
  //   initializer_list<float> value({1.0f ,1.0f ,1.0f ,1.0f});
  auto c = ops::Const(root, {{1.f, 1.f}, {1.f, 1.f}});
  auto s = ops::Const(root, 1.f);
  auto d = ops::Const(root, {{1.f, 1.f}, {1.f, 1.f}});

  auto add = ops::Add(root, var, c);

  auto assign = ops::Assign(root, var, add);

  auto apply_gradient_descent = ops::ApplyGradientDescent(root, var,s,d);

  // Turn off optimizations so that all the nodes are processed
  tensorflow::SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(tensorflow::OptimizerOptions_Level_L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(tensorflow::RewriterConfig::OFF);

  std::cout
      << "Currently selected backend: "
      << tensorflow::ngraph_bridge::BackendManager::GetCurrentlySetBackendName()
      << std::endl;

  ClientSession session(root, options);
  std::vector<tensorflow::Tensor> outputs;

  session.Run(
      {
          var_assign,
      },
      &outputs);

  std::cout << "initialize var: " << outputs[0].matrix<float>() << std::endl;

  for (int i = 0; i < 10; i++) {
    session.Run({assign}, &outputs);
    // Print the output,
    // right now prints out the TF tensor
    std::cout << "itr: " << i << " ,Result: " << outputs[0].matrix<float>()
             << std::endl;
  }
  session.Run({apply_gradient_descent}, &outputs);

 // this apply_gradient_descent result should be {10.0,10.0},{10.0,10.0}}
 std::cout << "ApplyGradientDescent value " << outputs[0].matrix<float>()
             << std::endl;

  session.Run({var}, &outputs);
  std::cout << "Final var: " << outputs[0].matrix<float>() << std::endl;

}

TEST(Variables, WeirdGraph2) {
  Scope root = Scope::NewRootScope();

  PartialTensorShape varShape({2, 2});
  auto var = ops::Variable(root.WithOpName("Var1"), varShape, DT_FLOAT);
  auto init_value = ops::Const(root, {{1.f, 1.f}, {1.f, 1.f}});
  auto var_assign = ops::Assign(root, var, init_value);

  //   TensorShape constShape({2,2});
  //   initializer_list<float> value({1.0f ,1.0f ,1.0f ,1.0f});
  auto c = ops::Const(root, {{1.f, 1.f}, {1.f, 1.f}});

  auto add = ops::Add(root.WithOpName("Add1"), var, c);

  auto assign = ops::Assign(root, var, add);

  auto add2 = ops::Add(root.WithOpName("Add2"), var, c);

  auto assign2 = ops::Assign(root, var, add2);

  // Turn off optimizations so that all the nodes are processed
  tensorflow::SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(tensorflow::OptimizerOptions_Level_L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(tensorflow::RewriterConfig::OFF);

  std::cout
      << "Currently selected backend: "
      << tensorflow::ngraph_bridge::BackendManager::GetCurrentlySetBackendName()
      << std::endl;

  ClientSession session(root, options);

  std::vector<tensorflow::Tensor> outputs;

  session.Run({var_assign},&outputs);
 std::cout << "initialize var: " << outputs[0].matrix<float>() << std::endl;
  for (int i = 0; i < 10; i++) {
    session.Run({assign2}, &outputs);
    // Print the output
    // right now print the output tensor of tf_tensor
    // std::cout << "itr: " << i << " ,Result: " << outputs[0].matrix<float>()
    //           << std::endl;
  }

  session.Run({var}, &outputs);
  //std::cout << "Final var: " << outputs[0].matrix<float>() << std::endl;

}

TEST(Variables, SmallGraph2) {
  Scope root = Scope::NewRootScope();

  PartialTensorShape varShape({2, 2});
  auto var = ops::Variable(root.WithOpName("Var1"), varShape, DT_FLOAT);
  auto init_value = ops::Const(root, {{1.f, 1.f}, {1.f, 1.f}});
  auto var_assign = ops::Assign(root, var, init_value);

  //   TensorShape constShape({2,2});
  //   initializer_list<float> value({1.0f ,1.0f ,1.0f ,1.0f});
  auto c = ops::Const(root, {{1.f, 1.f}, {1.f, 1.f}});

  auto add = ops::Add(root.WithOpName("Add1"), var, c);

  auto assign = ops::Assign(root, var, add);

  auto add2 = ops::Add(root.WithOpName("Add2"), var, c);

  auto assign2 = ops::Assign(root, var, add2);

  // Turn off optimizations so that all the nodes are processed
  tensorflow::SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(tensorflow::OptimizerOptions_Level_L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(tensorflow::RewriterConfig::OFF);

  std::cout
      << "Currently selected backend: "
      << tensorflow::ngraph_bridge::BackendManager::GetCurrentlySetBackendName()
      << std::endl;

  ClientSession session(root, options);

  std::vector<tensorflow::Tensor> outputs;

  session.Run({var_assign},&outputs);
 std::cout << "initialize var: " << outputs[0].matrix<float>() << std::endl;
  for (int i = 0; i < 10; i++) {
    session.Run({assign2}, &outputs);
    // Print the output
    std::cout << "itr: " << i << " ,Result: " << outputs[0].matrix<float>()
              << std::endl;
  }

  session.Run({var}, &outputs);
  std::cout << "Final var: " << outputs[0].matrix<float>() << std::endl;

}









}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow