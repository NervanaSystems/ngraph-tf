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
#include "ngraph_builder.h"
#include "ngraph_utils.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

#include "ngraph_backend_manager.h"

using namespace std;

tensorflow::Status SetNGraphBackend(const string& backend_name) {
  // Set the nGraph backend
  auto supported_backends =
      tensorflow::ngraph_bridge::BackendManager::GetSupportedBackendNames();
  vector<string> backends(supported_backends.begin(), supported_backends.end());

  for (auto& backend_name : backends) {
    cout << "Backend: " << backend_name << std::endl;
  }

  // Select a backend
  tensorflow::Status status =
      tensorflow::ngraph_bridge::BackendManager::SetBackendName(backend_name);
  return status;
}

// Run a simple TF op from a constructed graph
void MatMulExample() {
  // Create the graph
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();

  // Matrix A = [3 2; -1 0]
  auto A = tensorflow::ops::Const(root, {{3.f, 2.f}, {-1.f, 0.f}});
  // Vector b = [3 5]
  auto b = tensorflow::ops::Const(root, {{3.f, 5.f}});
  // v = Ab^T
  auto v = tensorflow::ops::MatMul(root.WithOpName("v"), A, b,
                                   tensorflow::ops::MatMul::TransposeB(true));

  std::vector<tensorflow::Tensor> outputs;

  tensorflow::ClientSession session(root);

  // Run and fetch v
  session.Run({v}, &outputs);

  std::cout
      << "Current backend: "
      << tensorflow::ngraph_bridge::BackendManager::GetCurrentlySetBackendName()
      << std::endl;

  // Expect outputs[0] == [19; -3]
  std::cout << "Result: " << outputs[0].matrix<float>() << std::endl;
}

int main(int argc, char** argv) {
  if (SetNGraphBackend("CPU") != tensorflow::Status::OK()) {
    std::cout << "Error: Cannot set the backend" << std::endl;
    return -1;
  }

  // Run the MatMul example
  MatMulExample();

  return 0;
}
