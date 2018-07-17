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
#include "gtest/gtest.h"

#include "ngraph_builder.h"
#include "ngraph_utils.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
using namespace std;
namespace tf = tensorflow;

namespace ngraph_bridge {

TEST(tf_exec, hello_world) {
  tf::Scope root = tf::Scope::NewRootScope();
  // Matrix A = [3 2; -1 0]
  auto A = tf::ops::Const(root, {{3.f, 2.f}, {-1.f, 0.f}});
  // Vector b = [3 5]
  auto b = tf::ops::Const(root, {{3.f, 5.f}});
  // v = Ab^T
  auto v = tf::ops::MatMul(root.WithOpName("v"), A, b,
                           tf::ops::MatMul::TransposeB(true));
  std::vector<tf::Tensor> outputs;
  tf::ClientSession session(root);
  // Run and fetch v
  TF_CHECK_OK(session.Run({v}, &outputs));
  // Expect outputs[0] == [19; -3]
  LOG(INFO) << outputs[0].matrix<float>();
}

TEST(tf_exec, axpy) {
  tf::GraphDef gdef;
  // auto status = tf::ReadTextProto(tf::Env::Default(), "test_py.pbtxt",
  // &gdef);
  auto status = tf::ReadTextProto(tf::Env::Default(), "test_axpy.pbtxt", &gdef);
  ASSERT_TRUE(status == tf::Status::OK()) << "Can't read protobuf graph";

  // tf::graph::SetDefaultDevice("/device:NGRAPH:0", &gdef);

  tf::SessionOptions options;
  tf::ConfigProto& config = options.config;
  config.set_allow_soft_placement(true);
  std::unique_ptr<tf::Session> session(tf::NewSession(options));

  TF_CHECK_OK(session->Create(gdef));

  // Create the inputs for this graph
  tf::Tensor x(tf::DT_FLOAT, tf::TensorShape({2, 3}));
  auto x_flat = x.flat<float>();
  for (int i = 0; i < x_flat.size(); i++) {
    x_flat.data()[i] = 1.0;
  }

  tf::Tensor y(tf::DT_FLOAT, tf::TensorShape({2, 3}));
  auto y_flat = y.flat<float>();
  for (int i = 0; i < y_flat.size(); i++) {
    y_flat.data()[i] = 1.0;
  }

  std::vector<tf::Tensor> outputs;

  TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {"mul", "add"}, {}, &outputs));

  ASSERT_EQ(outputs.size(), 2);
  auto mat1 = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(5.0, mat1(0, 0));
  EXPECT_FLOAT_EQ(5.0, mat1(1, 0));

  auto mat2 = outputs[1].matrix<float>();
  EXPECT_FLOAT_EQ(6.0, mat2(0, 0));
  EXPECT_FLOAT_EQ(6.0, mat2(1, 0));

  for (auto output : outputs) {
    auto output_flat = output.flat<float>();
    for (int i = 0; i < x_flat.size(); i++) {
      cout << output_flat.data()[i] << " ";
    }
    cout << endl;
  }
}

void AssertTensorEquals(tf::Tensor T1, tf::Tensor T2) {
  auto T_size = T1.flat<float>().size();
  for (int k=0; k<T_size; k++) {
    auto a = T1.flat<float>().data()[k];
    auto b = T2.flat<float>().data()[k];
    EXPECT_FLOAT_EQ(a, b);
  } 
}

TEST(tf_exec, BatchMatMul) { 
  tf::Scope root = tf::Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  auto A = tf::ops::Const(root, {-1.f, 2.f, 3.f, 4.f, -1.f, 2.f, 3.f, 4.f}, tf::TensorShape({2,2,2,1})); 
  auto B = tf::ops::Const(root, {1.f, 0.f, -1.f, -2.f, -1.f, 2.f, 3.f, 4.f}, tf::TensorShape({2,2,1,2})); 
  //auto X = tf::ops::Const(root, {1.f, 0.f, -1.f, -2.f, -1.f,1.f, 0.f, -1.f, -2.f, -2.f, -1.f,1.f, 0.f, -1.f, -2.f, -1.f, 1.f, 0.f, -1.f, -2.f, -1.f, 1.f, 0.f, -1.f}, tf::TensorShape({1,2,3,4}));
  //auto Y = tf::ops::Const(root, {1.f, 0.f, -1.f, -2.f, -1.f,1.f, 0.f, -1.f, -2.f, -2.f, -1.f,1.f, 0.f, -1.f, -2.f, -1.f, 1.f, 0.f, -1.f, -2.f, -1.f, 1.f, 0.f, -1.f}, tf::TensorShape({1,2,3,4}));
  tf::Tensor X1(tf::DT_FLOAT, tf::TensorShape({2, 3, 4, 5}));
  auto X1_flat = X1.flat<float>();
  for (int i = 0; i < X1_flat.size(); i++) {
    X1_flat.data()[i] = -1.1f*i;
  }
  tf::Tensor Y1(tf::DT_FLOAT, tf::TensorShape({2, 3, 4, 5}));
  auto Y1_flat = Y1.flat<float>();
  for (int i = 0; i < Y1_flat.size(); i++) {
    Y1_flat.data()[i] = -0.5f*i;
  }

  tf::Tensor X2(tf::DT_FLOAT, tf::TensorShape({2, 0, 4, 5}));
  auto X2_flat = X2.flat<float>();
  for (int i = 0; i < X2_flat.size(); i++) {
    X2_flat.data()[i] = -1.1f*i;
  }
  tf::Tensor Y2(tf::DT_FLOAT, tf::TensorShape({2, 0, 4, 5}));
  auto Y2_flat = Y2.flat<float>();
  for (int i = 0; i < Y2_flat.size(); i++) {
    Y2_flat.data()[i] = -0.5f*i;
  }
  tf::Tensor X3(tf::DT_FLOAT, tf::TensorShape({2, 3, 0, 5}));
  auto X3_flat = X3.flat<float>();
  for (int i = 0; i < X3_flat.size(); i++) {
    X3_flat.data()[i] = -1.1f*i;
  }
  tf::Tensor Y3(tf::DT_FLOAT, tf::TensorShape({2, 3, 0, 5}));
  auto Y3_flat = Y3.flat<float>();
  for (int i = 0; i < Y3_flat.size(); i++) {
    Y3_flat.data()[i] = -0.5f*i;
  }


  auto R = tf::ops::BatchMatMul(dev_scope.WithOpName("R"), A, B);
  auto attrs = tf::ops::BatchMatMul::Attrs().AdjX(true);
  auto attrs_y = tf::ops::BatchMatMul::Attrs().AdjY(true); 
  //bool tensorflow::ops::BatchMatMul::Attrs adj_y = true;
  auto Z1 = tf::ops::BatchMatMul(dev_scope.WithOpName("Z1"), X1, Y1, attrs);
  auto Z2 = tf::ops::BatchMatMul(dev_scope.WithOpName("Z2"), X2, Y2, attrs);
  auto Z3 = tf::ops::BatchMatMul(dev_scope.WithOpName("Z3"), X3, Y3, attrs);
  auto Z4 = tf::ops::BatchMatMul(dev_scope.WithOpName("Z4"), X3, Y3, attrs_y);
  std::vector<tf::Tensor> outputs;
  std::vector<tf::Tensor> outputs_z1;
  std::vector<tf::Tensor> outputs_z2;
  std::vector<tf::Tensor> outputs_z3;
  std::vector<tf::Tensor> outputs_z4;
  // Run and fetch v
  tf::ClientSession session(dev_scope);
  TF_CHECK_OK(session.Run({R}, &outputs));
  TF_CHECK_OK(session.Run({Z1}, &outputs_z1));
  TF_CHECK_OK(session.Run({Z2}, &outputs_z2)); 
  TF_CHECK_OK(session.Run({Z3}, &outputs_z3)); 
  TF_CHECK_OK(session.Run({Z4}, &outputs_z4));
  // Expect outputs[0] == [19; -3]

  tf::ClientSession sess(root);
  std::vector<tf::Tensor> outputs_cpu;
  std::vector<tf::Tensor> outputs_z1_cpu;
  std::vector<tf::Tensor> outputs_z2_cpu;
  std::vector<tf::Tensor> outputs_z3_cpu;
  std::vector<tf::Tensor> outputs_z4_cpu;
  auto C = tf::ops::BatchMatMul(root.WithOpName("C"), A, B);
  auto W1 = tf::ops::BatchMatMul(root.WithOpName("W1"), X1, Y1, attrs); 
  auto W2 = tf::ops::BatchMatMul(root.WithOpName("W2"), X2, Y2, attrs);
  auto W3 = tf::ops::BatchMatMul(root.WithOpName("W3"), X3, Y3, attrs);
  auto W4 = tf::ops::BatchMatMul(root.WithOpName("W4"), X3, Y3, attrs_y);
  TF_CHECK_OK(sess.Run({C}, &outputs_cpu));
  TF_CHECK_OK(sess.Run({W1}, &outputs_z1_cpu));
  TF_CHECK_OK(sess.Run({W2}, &outputs_z2_cpu));
  TF_CHECK_OK(sess.Run({W3}, &outputs_z3_cpu));
  TF_CHECK_OK(sess.Run({W4}, &outputs_z4_cpu));
  ASSERT_EQ(outputs[0].shape(),outputs_cpu[0].shape());
  ASSERT_EQ(outputs_z1[0].shape(),outputs_z1_cpu[0].shape());
  ASSERT_EQ(outputs_z2[0].shape(),outputs_z2_cpu[0].shape());
  ASSERT_EQ(outputs_z3[0].shape(),outputs_z3_cpu[0].shape());
  ASSERT_EQ(outputs_z4[0].shape(),outputs_z4_cpu[0].shape());
  AssertTensorEquals(outputs_z1[0],outputs_z1_cpu[0]);
  AssertTensorEquals(outputs_z2[0],outputs_z2_cpu[0]); 
  AssertTensorEquals(outputs_z3[0],outputs_z3_cpu[0]); 
  AssertTensorEquals(outputs_z4[0],outputs_z4_cpu[0]);
}

TEST(tf_exec, BatchMatMul_3D) { 
  tf::Scope root = tf::Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  auto A = tf::ops::Const(root, {-1.f, 2.f, 3.f, 4.f, -1.f, 2.f, 3.f, 4.f}, tf::TensorShape({2,2,2})); 
  auto B = tf::ops::Const(root, {1.f, 0.f, -1.f, -2.f, -1.f, 2.f, 3.f, 4.f}, tf::TensorShape({2,2,2})); 
  auto R = tf::ops::BatchMatMul(dev_scope.WithOpName("R"), A, B);
  std::vector<tf::Tensor> outputs;
  // Run and fetch v
  tf::ClientSession session(dev_scope);
  TF_CHECK_OK(session.Run({R}, &outputs));
  // Expect outputs[0] == [19; -3]
  ASSERT_EQ(outputs[0].shape(), tf::TensorShape({2,2,2}));

  tf::ClientSession sess(root);
  std::vector<tf::Tensor> outputs_cpu;
  auto C = tf::ops::BatchMatMul(root.WithOpName("C"), A, B);
  TF_CHECK_OK(sess.Run({C}, &outputs_cpu));
  ASSERT_EQ(outputs[0].shape(),outputs_cpu[0].shape());
  AssertTensorEquals(outputs[0],outputs_cpu[0]);
}

TEST(tf_exec, BatchMatMul_2D) { 
  tf::Scope root = tf::Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  auto A = tf::ops::Const(root, {-1.f, 2.f, 3.f, 4.f}, tf::TensorShape({2,2})); 
  auto B = tf::ops::Const(root, {1.f, 0.f, -1.f, -2.f}, tf::TensorShape({2,2})); 
  auto R = tf::ops::BatchMatMul(dev_scope.WithOpName("R"), A, B);
  std::vector<tf::Tensor> outputs;
  // Run and fetch R
  tf::ClientSession session(dev_scope);
  TF_CHECK_OK(session.Run({R}, &outputs));
  // Expect outputs[0] == [19; -3]
  ASSERT_EQ(outputs[0].shape(), tf::TensorShape({2,2}));
  auto mat = outputs[0].matrix<float>();
  ASSERT_EQ(-3.f, mat(0,0));
  ASSERT_EQ(-4.f, mat(0,1)); 
  ASSERT_EQ(-1.f, mat(1,0));
  ASSERT_EQ(-8.f, mat(1,1));

  tf::ClientSession sess(root);
  std::vector<tf::Tensor> outputs_cpu;
  auto C = tf::ops::BatchMatMul(root.WithOpName("C"), A, B);
  TF_CHECK_OK(sess.Run({C}, &outputs_cpu));
  ASSERT_EQ(outputs[0].shape(),outputs_cpu[0].shape());
  AssertTensorEquals(outputs[0],outputs_cpu[0]);
}


}  // namespace ngraph_bridge
