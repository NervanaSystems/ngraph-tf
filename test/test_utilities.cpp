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

#include "test_utilities.h"
#include <cstdlib>
#include <ctime>

using namespace std;

namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

void ActivateNGraph() {
  setenv("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS", "1", 1);
  unsetenv("NGRAPH_TF_DISABLE");
}

void DeactivateNGraph() {
  unsetenv("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS");
  setenv("NGRAPH_TF_DISABLE", "1", 1);
}

void AssignInputIntValues(Tensor& A, int maxval) {
  auto A_flat = A.flat<int>();
  auto A_flat_data = A_flat.data();
  int counter = 0;
  for (int i = 0; i < A_flat.size(); i++) {
    A_flat_data[i] = counter++;
    if (counter == maxval) {
      counter = 0;
    }
  }
}

void AssignInputValues(Tensor& A, float x) {
  auto A_flat = A.flat<float>();
  auto A_flat_data = A_flat.data();
  for (int i = 0; i < A_flat.size(); i++) {
    A_flat_data[i] = x;
  }
}

// Input x will be used as an anchor
// Actual value assigned equals to x * i
void AssignInputValuesAnchor(Tensor& A, float x) {
  auto A_flat = A.flat<float>();
  auto A_flat_data = A_flat.data();
  for (int i = 0; i < A_flat.size(); i++) {
    A_flat_data[i] = x * i;
  }
}

// Randomly generate a float number between -10.00 ~ 10.99
void AssignInputValuesRandom(Tensor& A) {
  auto A_flat = A.flat<float>();
  auto A_flat_data = A_flat.data();
  srand(static_cast<unsigned>(time(0)));
  for (int i = 0; i < A_flat.size(); i++) {
    // give a number between 0 and 20
    float value =
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX / 20.0f);
    value = (value - 10.0f);  // range from -10 to 10
    value =
        roundf(value * 100) / 100.0;  // change the precision of the float to
                                      // 2 number after the decimal
    A_flat_data[i] = value;
  }
}

void PrintTensor(const Tensor& T1) {
  LOG(INFO) << "print tensor values" << T1.DebugString();
}

void ValidateTensorData(Tensor& T1, Tensor& T2, float tol) {
  ASSERT_EQ(T1.shape(), T2.shape());
  auto T_size = T1.flat<float>().size();
  auto T1_data = T1.flat<float>().data();
  auto T2_data = T2.flat<float>().data();
  for (int k = 0; k < T_size; k++) {
    auto a = T1_data[k];
    auto b = T2_data[k];
    if (a == 0) {
      EXPECT_NEAR(a, b, tol);
    } else {
      auto rel = a - b;
      auto rel_div = std::abs(rel / a);
      EXPECT_TRUE(rel_div < tol);
    }
  }
}

template <>
bool eq(float arg0, float arg1) {
  if (arg0 == 0 && arg1 == 0) {
    return true;
  } else {
    return (abs(arg0 - arg1) / max(abs(arg0), abs(arg1)) <= 0.001);
  }
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
