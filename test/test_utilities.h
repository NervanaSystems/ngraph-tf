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
#ifndef NGRAPH_TF_BRIDGE_TESTUTILITIES_H_
#define NGRAPH_TF_BRIDGE_TESTUTILITIES_H_

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/env.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {
// some utility functions copied from tf_exec.cpp
void ActivateNGraph();
void DeactivateNGraph();
void AssertTensorEqualsFloat(Tensor& T1, Tensor& T2);
void AssertTensorEqualsInt32(Tensor& T1, Tensor& T2);
void AssignInputIntValues(Tensor& A, int maxval);
void AssignInputValues(Tensor& A, float x);
void AssignInputValuesAnchor(Tensor& A, float x);  // value assigned = x * index
void AssignInputValuesRandom(Tensor& A);
void PrintTensor(const Tensor& T1);
void ValidateTensorData(Tensor& T1, Tensor& T2, float tol);

template <typename T>
void AssignInputValuesFromVector(Tensor& A, vector<T> x) {
  auto A_flat = A.flat<T>();
  auto A_flat_data = A_flat.data();
  assert(A_flat.size() == x.size());
  for (int i = 0; i < A_flat.size(); i++) {
    A_flat_data[i] = x[i];
  }
}

// Randomly generate data with specified type to populate the Tensor
// Random data is generated within range (min, max)
template <typename T>
void AssignInputValuesRandom(Tensor& A, T min, T max) {
  auto A_flat = A.flat<T>();
  auto A_flat_data = A_flat.data();
  srand(static_cast<unsigned>(time(0)));
  for (int i = 0; i < A_flat.size(); i++) {
    T value =
        // randomly generate a number between 0 and (max-min) inclusive
        static_cast<T>(rand()) / static_cast<T>(RAND_MAX / (max - min + 1));
    value = value + min;  // transform the range to (min, max) inclusive
    A_flat_data[i] = value;
  }
}

template <class T>
bool eq(T arg0, T arg1) {
  return arg0 == arg1;
}

template <typename T>
void AssertTensorEquals(const Tensor& T1, const Tensor& T2) {
  ASSERT_EQ(T1.shape(), T2.shape());
  ASSERT_EQ(T1.dtype(), T2.dtype());
  auto T_size = T1.flat<T>().size();
  auto T1_data = T1.flat<T>().data();
  auto T2_data = T2.flat<T>().data();
  for (int k = 0; k < T_size; k++) {
    auto a = T1_data[k];
    auto b = T2_data[k];
    bool rt = eq<T>(a, b);
    EXPECT_TRUE(rt) << " TF output " << a << endl << " NG output " << b;
  }
}

}  // namespace ngraph_bridge

}  // namespace tensorflow

#endif  // NGRAPH_TF_BRIDGE_TESTUTILITIES_H_
