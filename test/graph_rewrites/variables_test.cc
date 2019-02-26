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

TEST(Variables, SmallGraph1) {
  Scope root = Scope::NewRootScope();

  PartialTensorShape varShape({2, 2});
  auto var = ops::Variable(root.WithOpName("Var1"), varShape, DT_FLOAT);
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

  ClientSession session(root, options);

  std::vector<tensorflow::Tensor> ng_output_init;
  std::vector<tensorflow::Tensor> ng_output_compute;
  std::vector<tensorflow::Tensor> ng_output_final;
  std::vector<tensorflow::Tensor> ng_output_compute2;

  // Initialize the Variable
  session.Run(
      {
          var_assign,
      },
      &ng_output_init);
  std::cout << "initialize var: " << ng_output_init[0].matrix<float>()
            << std::endl;

  // Update the Variable
  for (int i = 0; i < 10; i++) {
    session.Run({assign}, &ng_output_compute);
    // Print the output
    std::cout << "itr: " << i
              << " ,Result: " << ng_output_compute[0].matrix<float>()
              << std::endl;
  }

  // Final Variable Value
  session.Run({var}, &ng_output_final);
  std::cout << "Final var: " << ng_output_final[0].matrix<float>() << std::endl;

  for (int i = 0; i < 10; i++) {
    session.Run({assign}, &ng_output_compute2);
    // Print the output
    std::cout << "itr: " << i
              << " ,Result: " << ng_output_compute2[0].matrix<float>()
              << std::endl;
  }

  //[TODO]Run the Graph on TF now and compare the values
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow