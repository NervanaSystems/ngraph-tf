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
#include <fstream>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph_timer.h"

#include <mutex>

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {
namespace testing {

#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());
#undef ASSERT_OK

// TEST(tf_exec, hello_world) {
//   Scope root = Scope::NewRootScope();

//   // root = root.WithDevice("/device:NGRAPH:0");
//   // Matrix A = [3 2; -1 0]
//   auto A = ops::Const(root, {{3.f, 2.f}, {-1.f, 0.f}});
//   // Vector b = [3 5]
//   auto b = ops::Const(root, {{3.f, 5.f}});
//   // v = Ab^T
//   auto v =
//       ops::MatMul(root.WithOpName("v"), A, b, ops::MatMul::TransposeB(true));
//   std::vector<Tensor> outputs;
//   ClientSession session(root);
//   // Run and fetch v
//   ASSERT_OK(session.Run({v}, &outputs));
//   // Expect outputs[0] == [19; -3]
//   LOG(INFO) << outputs[0].matrix<float>();
// }

TEST(timer, event_record) {
  // Create a json file
  std::ofstream trace_file("test_events.json");
  Event evt_1("Event 1", "Testing");

  trace_file << "[" << std::endl;
  std::vector<std::thread> threads;
  std::mutex mtx;
  for (auto i = 0; i < 10; i++) {
    int id = i;
    std::thread next_thread([&] {
      std::ostringstream oss;
      oss << "Event: " << id;
      Event event(oss.str().c_str(), "Dummy");
      std::cout << "Thread: " << i << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      event.Stop();
      std::lock_guard<std::mutex> lock(mtx);
      trace_file << event << "," << std::endl;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    threads.push_back(std::move(next_thread));
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  evt_1.Stop();

  for (auto& next : threads) {
    next.join();
  }

  trace_file << "]" << std::endl;
}

}  // namespace testing

}  // namespace ngraph_bridge

}  // namespace tensorflow
