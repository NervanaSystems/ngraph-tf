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
#ifndef NGRAPH_TF_BRIDGE_TestUtilities_H_
#define NGRAPH_TF_BRIDGE_TestUtilities_H_

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
void DummyActivateNGraph();
void DummyDeactivateNGraph();
void DummyAssertTensorEquals(Tensor& T1, Tensor& T2);
void DummyAssignInputIntValues(Tensor& A, int maxval);
void DummyAssignInputValues(Tensor& A, float x);
void DummyPrintTensor(const Tensor& T1);
Status NGraphShapeToTFShape(ng::Shape& ng_op_shape, TensorShape *tf_shape);

}  // namespace ngraph_bridge

}  // namespace tensorflow

#endif //NGRAPH_TF_BRIDGE_TestUtilities_H_
