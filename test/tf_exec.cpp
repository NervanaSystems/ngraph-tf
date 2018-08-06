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

namespace tensorflow {

namespace ngraph_bridge {

#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());

TEST(tf_exec, hello_world) {
  Scope root = Scope::NewRootScope();

  // root = root.WithDevice("/device:NGRAPH:0");
  // Matrix A = [3 2; -1 0]
  auto A = ops::Const(root, {{3.f, 2.f}, {-1.f, 0.f}});
  // Vector b = [3 5]
  auto b = ops::Const(root, {{3.f, 5.f}});
  // v = Ab^T
  auto v =
      ops::MatMul(root.WithOpName("v"), A, b, ops::MatMul::TransposeB(true));
  std::vector<Tensor> outputs;
  ClientSession session(root);
  // Run and fetch v
  ASSERT_OK(session.Run({v}, &outputs));
  // Expect outputs[0] == [19; -3]
  LOG(INFO) << outputs[0].matrix<float>();
}

#if !defined(NGRAPH_EMBEDDED_IN_TENSORFLOW)
TEST(tf_exec, axpy) {
  GraphDef gdef;
  // auto status = ReadTextProto(Env::Default(), "test_py.pbtxt",
  // &gdef);
  auto status = ReadTextProto(Env::Default(), "test_axpy.pbtxt", &gdef);
  ASSERT_TRUE(status == Status::OK()) << "Can't read protobuf graph";

  // graph::SetDefaultDevice("/device:NGRAPH:0", &gdef);

  SessionOptions options;
  ConfigProto& config = options.config;
  config.set_allow_soft_placement(true);
  std::unique_ptr<Session> session(NewSession(options));

  ASSERT_OK(session->Create(gdef));

  // Create the inputs for this graph
  Tensor x(DT_FLOAT, TensorShape({2, 3}));
  auto x_flat = x.flat<float>();
  for (int i = 0; i < x_flat.size(); i++) {
    x_flat.data()[i] = 1.0;
  }

  Tensor y(DT_FLOAT, TensorShape({2, 3}));
  auto y_flat = y.flat<float>();
  for (int i = 0; i < y_flat.size(); i++) {
    y_flat.data()[i] = 1.0;
  }

  std::vector<Tensor> outputs;

  ASSERT_OK(session->Run({{"x", x}, {"y", y}}, {"mul", "add"}, {}, &outputs));

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
#endif

void AssertTensorEquals(Tensor T1, Tensor T2) {
  auto T_size = T1.flat<float>().size();
  for (int k = 0; k < T_size; k++) {
    auto a = T1.flat<float>().data()[k];
    auto b = T2.flat<float>().data()[k];
    EXPECT_FLOAT_EQ(a, b);
  }
}

TEST(tf_exec, BatchMatMul_0D) {
  Scope root = Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");

  Tensor X1(DT_FLOAT, TensorShape({2, 0, 4, 5}));
  Tensor Y1(DT_FLOAT, TensorShape({2, 0, 4, 5}));
  Tensor X2(DT_FLOAT, TensorShape({2, 3, 0, 5}));
  Tensor Y2(DT_FLOAT, TensorShape({2, 3, 0, 5}));

  auto attrs_x = ops::BatchMatMul::Attrs().AdjX(true);
  auto attrs_y = ops::BatchMatMul::Attrs().AdjY(true);
  auto Z1 = ops::BatchMatMul(dev_scope.WithOpName("Z1"), X1, Y1, attrs_x);
  auto Z2 = ops::BatchMatMul(dev_scope.WithOpName("Z2"), X2, Y2, attrs_x);
  auto Z = ops::BatchMatMul(dev_scope.WithOpName("Z"), X2, Y2, attrs_y);
  std::vector<Tensor> outputs_z1;
  std::vector<Tensor> outputs_z2;
  std::vector<Tensor> outputs_z;
  // Run and fetch v
  ClientSession session(dev_scope);
  ASSERT_OK(session.Run({Z1}, &outputs_z1));
  ASSERT_OK(session.Run({Z2}, &outputs_z2));
  ASSERT_OK(session.Run({Z}, &outputs_z));
  // Expect outputs[0] == [19; -3]

  ClientSession sess(root);
  std::vector<Tensor> outputs_z1_cpu;
  std::vector<Tensor> outputs_z2_cpu;
  std::vector<Tensor> outputs_z_cpu;
  auto W1 = ops::BatchMatMul(root.WithOpName("W1"), X1, Y1, attrs_x);
  auto W2 = ops::BatchMatMul(root.WithOpName("W2"), X2, Y2, attrs_x);
  auto W = ops::BatchMatMul(root.WithOpName("W"), X2, Y2, attrs_y);
  ASSERT_OK(sess.Run({W1}, &outputs_z1_cpu));
  ASSERT_OK(sess.Run({W2}, &outputs_z2_cpu));
  ASSERT_OK(sess.Run({W}, &outputs_z_cpu));
  ASSERT_EQ(outputs_z1[0].shape(), outputs_z1_cpu[0].shape());
  ASSERT_EQ(outputs_z2[0].shape(), outputs_z2_cpu[0].shape());
  ASSERT_EQ(outputs_z[0].shape(), outputs_z_cpu[0].shape());
  AssertTensorEquals(outputs_z1[0], outputs_z1_cpu[0]);
  AssertTensorEquals(outputs_z2[0], outputs_z2_cpu[0]);
  AssertTensorEquals(outputs_z[0], outputs_z_cpu[0]);
}

void ActivateNGraph() {
  setenv("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS", "1", 1);
  unsetenv("NGRAPH_TF_DISABLE");
}

void DeactivateNGraph() {
  unsetenv("NGRAPH_TF_DISABLE_DEASSIGN_CLUSTERS");
  setenv("NGRAPH_TF_DISABLE", "1", 1);
}

TEST(tf_exec, BatchMatMul) {
  Scope root = Scope::NewRootScope();

  auto A = ops::Const(root, {-1.f, 2.f, 3.f, 4.f, -1.f, 2.f, 3.f, 4.f},
                      TensorShape({2, 2, 2, 1}));
  auto B = ops::Const(root, {1.f, 0.f, -1.f, -2.f, -1.f, 2.f, 3.f, 4.f},
                      TensorShape({2, 2, 1, 2}));

  Tensor X(DT_FLOAT, TensorShape({2, 3, 4, 5}));
  auto X_flat = X.flat<float>();
  for (int i = 0; i < X_flat.size(); i++) {
    X_flat.data()[i] = -1.1f * i;
  }
  Tensor Y(DT_FLOAT, TensorShape({2, 3, 4, 5}));
  auto Y_flat = Y.flat<float>();
  for (int i = 0; i < Y_flat.size(); i++) {
    Y_flat.data()[i] = -0.5f * i;
  }

  // Run on nGraph
  auto R = ops::BatchMatMul(root.WithOpName("R"), A, B);
  auto attrs_x = ops::BatchMatMul::Attrs().AdjX(true);
  auto attrs_y = ops::BatchMatMul::Attrs().AdjY(true);
  auto Z1 = ops::BatchMatMul(root.WithOpName("Z1"), X, Y, attrs_x);
  auto Z2 = ops::BatchMatMul(root.WithOpName("Z2"), X, Y, attrs_y);

  std::vector<Tensor> outputs_ng;
  std::vector<Tensor> outputs_z1_ng;
  std::vector<Tensor> outputs_z2_ng;

  ActivateNGraph();
  ClientSession session_ng(root);
  ASSERT_OK(session_ng.Run({R}, &outputs_ng));
  ASSERT_OK(session_ng.Run({Z1}, &outputs_z1_ng));
  ASSERT_OK(session_ng.Run({Z2}, &outputs_z2_ng));

  std::vector<Tensor> outputs_tf;
  std::vector<Tensor> outputs_z1_tf;
  std::vector<Tensor> outputs_z2_tf;

  DeactivateNGraph();
  ClientSession session_tf(root);
  ASSERT_OK(session_tf.Run({R}, &outputs_tf));
  ASSERT_OK(session_tf.Run({Z1}, &outputs_z1_tf));
  ASSERT_OK(session_tf.Run({Z2}, &outputs_z2_tf));

  // Check results for equality
  ASSERT_EQ(outputs_ng[0].shape(), outputs_tf[0].shape());
  ASSERT_EQ(outputs_z1_ng[0].shape(), outputs_z1_tf[0].shape());
  ASSERT_EQ(outputs_z2_ng[0].shape(), outputs_z2_tf[0].shape());
  AssertTensorEquals(outputs_z1_ng[0], outputs_z1_tf[0]);
  AssertTensorEquals(outputs_z2_ng[0], outputs_z2_tf[0]);
}

TEST(tf_exec, BatchMatMul_3D) {
  Scope root = Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  auto A = ops::Const(root, {-1.f, 2.f, 3.f, 4.f, -1.f, 2.f, 3.f, 4.f},
                      TensorShape({2, 2, 2}));
  auto B = ops::Const(root, {1.f, 0.f, -1.f, -2.f, -1.f, 2.f, 3.f, 4.f},
                      TensorShape({2, 2, 2}));
  auto R = ops::BatchMatMul(dev_scope.WithOpName("R"), A, B);
  Tensor X(DT_FLOAT, TensorShape({2, 3, 4}));
  auto X_flat = X.flat<float>();
  for (int i = 0; i < X_flat.size(); i++) {
    X_flat.data()[i] = -1.1f * i;
  }
  Tensor Y(DT_FLOAT, TensorShape({2, 3, 4}));
  auto Y_flat = Y.flat<float>();
  for (int i = 0; i < Y_flat.size(); i++) {
    Y_flat.data()[i] = -0.5f * i;
  }

  auto attrs_x = ops::BatchMatMul::Attrs().AdjX(true);
  auto attrs_y = ops::BatchMatMul::Attrs().AdjY(true);
  auto Z1 = ops::BatchMatMul(dev_scope.WithOpName("Z1"), X, Y, attrs_x);
  auto Z2 = ops::BatchMatMul(dev_scope.WithOpName("Z2"), X, Y, attrs_y);
  std::vector<Tensor> outputs;
  std::vector<Tensor> outputs_z1;
  std::vector<Tensor> outputs_z2;
  // Run and fetch v
  ClientSession session(dev_scope);
  ASSERT_OK(session.Run({R}, &outputs));
  ASSERT_OK(session.Run({Z1}, &outputs_z1));
  ASSERT_OK(session.Run({Z2}, &outputs_z2));
  // Expect outputs[0] == [19; -3]

  ClientSession sess(root);
  std::vector<Tensor> outputs_cpu;
  std::vector<Tensor> outputs_z1_cpu;
  std::vector<Tensor> outputs_z2_cpu;
  auto C = ops::BatchMatMul(root.WithOpName("C"), A, B);
  auto W1 = ops::BatchMatMul(root.WithOpName("W1"), X, Y, attrs_x);
  auto W2 = ops::BatchMatMul(root.WithOpName("W2"), X, Y, attrs_y);
  ASSERT_OK(sess.Run({C}, &outputs_cpu));
  ASSERT_OK(sess.Run({W1}, &outputs_z1_cpu));
  ASSERT_OK(sess.Run({W2}, &outputs_z2_cpu));
  ASSERT_EQ(outputs[0].shape(), outputs_cpu[0].shape());
  ASSERT_EQ(outputs_z1[0].shape(), outputs_z1_cpu[0].shape());
  ASSERT_EQ(outputs_z2[0].shape(), outputs_z2_cpu[0].shape());
  AssertTensorEquals(outputs_z1[0], outputs_z1_cpu[0]);
  AssertTensorEquals(outputs_z2[0], outputs_z2_cpu[0]);
}

TEST(tf_exec, BatchMatMul_2D) {
  Scope root = Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  auto A = ops::Const(root, {-1.f, 2.f, 3.f, 4.f}, TensorShape({2, 2}));
  auto B = ops::Const(root, {1.f, 0.f, -1.f, -2.f}, TensorShape({2, 2}));
  auto R = ops::BatchMatMul(dev_scope.WithOpName("R"), A, B);
  std::vector<Tensor> outputs;
  // Run and fetch R
  ClientSession session(dev_scope);
  ASSERT_OK(session.Run({R}, &outputs));
  // Expect outputs[0] == [19; -3]
  auto mat = outputs[0].matrix<float>();
  ASSERT_EQ(-3.f, mat(0, 0));
  ASSERT_EQ(-4.f, mat(0, 1));
  ASSERT_EQ(-1.f, mat(1, 0));
  ASSERT_EQ(-8.f, mat(1, 1));

  ClientSession sess(root);
  std::vector<Tensor> outputs_cpu;
  auto C = ops::BatchMatMul(root.WithOpName("C"), A, B);
  ASSERT_OK(sess.Run({C}, &outputs_cpu));
  ASSERT_EQ(outputs[0].shape(), outputs_cpu[0].shape());
  AssertTensorEquals(outputs[0], outputs_cpu[0]);
}

TEST(tf_exec, FusedBatchNormGrad_NHWC) {
  Scope root = Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  Tensor tf_input(DT_FLOAT, TensorShape({5, 3, 4, 2}));
  Tensor tf_delta(DT_FLOAT, TensorShape({5, 3, 4, 2}));
  Tensor tf_mean(DT_FLOAT, TensorShape({2}));
  Tensor tf_variance(DT_FLOAT, TensorShape({2}));
  Tensor tf_gamma(DT_FLOAT, TensorShape({2}));

  auto tf_input_flat = tf_input.flat<float>();
  for (int i = 0; i < tf_input_flat.size(); i++) {
    tf_input_flat.data()[i] = -1.1f * i;
  }
  auto tf_delta_flat = tf_delta.flat<float>();
  for (int i = 0; i < tf_delta_flat.size(); i++) {
    tf_delta_flat.data()[i] = -2.1f * i;
  }
  auto tf_mean_flat = tf_mean.flat<float>();
  for (int i = 0; i < tf_mean_flat.size(); i++) {
    tf_mean_flat.data()[i] = 1.1f * i;
  }
  auto tf_variance_flat = tf_variance.flat<float>();
  for (int i = 0; i < tf_variance_flat.size(); i++) {
    tf_variance_flat.data()[i] = 0.5f * i;
  }
  auto tf_gamma_flat = tf_gamma.flat<float>();
  for (int i = 0; i < tf_gamma_flat.size(); i++) {
    tf_gamma_flat.data()[i] = -1.6f * i;
  }

  auto attrs = ops::FusedBatchNormGrad::Attrs();
  attrs.is_training_ = true;
  attrs.epsilon_ = 0.0001f;
  attrs.data_format_ = "NHWC";

  std::vector<Tensor> outputs;
  ClientSession session(dev_scope);
  auto R =
      ops::FusedBatchNormGrad(dev_scope.WithOpName("R"), tf_delta, tf_input,
                              tf_gamma, tf_mean, tf_variance, attrs);
  ASSERT_OK(session.Run({R.x_backprop, R.scale_backprop, R.offset_backprop},
                        &outputs));

  ClientSession sess(root);
  std::vector<Tensor> outputs_cpu;
  auto C = ops::FusedBatchNormGrad(root.WithOpName("C"), tf_delta, tf_input,
                                   tf_gamma, tf_mean, tf_variance, attrs);
  ASSERT_OK(sess.Run({C.x_backprop, C.scale_backprop, C.offset_backprop},
                     &outputs_cpu));
  ASSERT_EQ(outputs[0].shape(), outputs_cpu[0].shape());
  ASSERT_EQ(outputs[1].shape(), outputs_cpu[1].shape());
  ASSERT_EQ(outputs[2].shape(), outputs_cpu[2].shape());
  AssertTensorEquals(outputs[0], outputs_cpu[0]);
  AssertTensorEquals(outputs[1], outputs_cpu[1]);
  AssertTensorEquals(outputs[2], outputs_cpu[2]);
}

TEST(tf_exec, Tile) {
  Scope root = Scope::NewRootScope();
  auto dev_scope = root.WithDevice("/device:NGRAPH:0");
  Tensor A(DT_FLOAT, TensorShape({2, 3, 4}));
  auto A_flat = A.flat<float>();
  for (int i = 0; i < A_flat.size(); i++) {
    A_flat.data()[i] = -1.1f * i;
  }
  auto X = ops::Const(root, {int64(3), int64(4), int64(2)}, TensorShape({3}));
  auto Y = ops::Const(root, {int64(1), int64(0), int64(3)}, TensorShape({3}));
  auto C = ops::Tile(dev_scope.WithOpName("C"), A, X);
  auto D = ops::Tile(dev_scope.WithOpName("D"), A, Y);
  std::vector<Tensor> outputs_C;
  std::vector<Tensor> outputs_D;

  ClientSession session(dev_scope);
  ASSERT_OK(session.Run({C}, &outputs_C));
  ASSERT_OK(session.Run({D}, &outputs_D));

  ClientSession sess(root);
  std::vector<Tensor> outputs_C_cpu;
  std::vector<Tensor> outputs_D_cpu;
  auto C_cpu = ops::Tile(root.WithOpName("C_cpu"), A, X);
  auto D_cpu = ops::Tile(root.WithOpName("D_cpu"), A, Y);
  ASSERT_OK(sess.Run({C_cpu}, &outputs_C_cpu));
  ASSERT_OK(sess.Run({D_cpu}, &outputs_D_cpu));
  ASSERT_EQ(outputs_C[0].shape(), outputs_C_cpu[0].shape());
  ASSERT_EQ(outputs_D[0].shape(), outputs_D_cpu[0].shape());
  AssertTensorEquals(outputs_C[0], outputs_C_cpu[0]);
  AssertTensorEquals(outputs_D[0], outputs_D_cpu[0]);
}

// Test Op :"Op_RealDiv"
// With Const inputs tensorflow's constant folding optimisation converts the op
// to "Mul".
// To test "RealDiv" operator, explicitly placed the op on NGRAPH and the inputs
// as placeholders
TEST(tf_exec, Op_RealDiv) {
  Scope root = Scope::NewRootScope();
  Scope root_ngraph = root.NewSubScope("sub_scope_ngraph");
  root_ngraph = root_ngraph.WithDevice("/device:NGRAPH:0");

  auto A = ops::Placeholder(root, DataType::DT_FLOAT);
  auto B = ops::Placeholder(root, DataType::DT_FLOAT);
  auto r = ops::RealDiv(root_ngraph.WithOpName("r"), A, B);

  std::vector<Tensor> outputs;
  ClientSession session(root);

  ASSERT_OK(session.Run(
      {{A, {{3.f, 5.f}, {2.f, 0.f}}}, {B, {{3.f, 2.f}, {.1f, 1.f}}}}, {r},
      &outputs));

  ASSERT_EQ(outputs[0].shape(), TensorShape({2, 2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(1.0, mat(0, 0));
  EXPECT_FLOAT_EQ(2.5, mat(0, 1));
  EXPECT_FLOAT_EQ(20.0, mat(1, 0));
  EXPECT_FLOAT_EQ(0.0, mat(1, 1));
}

TEST(tf_exec, Op_Reciprocal) {
  Scope root = Scope::NewRootScope();
  Scope root_ngraph = root.NewSubScope("sub_scope_ngraph");
  root_ngraph = root_ngraph.WithDevice("/device:NGRAPH:0");

  auto A = ops::Placeholder(root, DataType::DT_FLOAT);
  auto r = ops::Reciprocal(root_ngraph.WithOpName("r"), A);

  std::vector<Tensor> outputs;
  ClientSession session(root);

  ASSERT_OK(session.Run({{A, {{1.f, 5.f}, {2.f, 1.f}}}}, {r}, &outputs));
  ASSERT_EQ(outputs[0].shape(), TensorShape({2, 2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(1.0, mat(0, 0));
  EXPECT_FLOAT_EQ(0.2, mat(0, 1));
  EXPECT_FLOAT_EQ(0.5, mat(1, 0));
  EXPECT_FLOAT_EQ(1.0, mat(1, 1));
}

TEST(tf_exec, Op_Square) {
  Scope root = Scope::NewRootScope();
  root = root.WithDevice("/device:NGRAPH:0");

  auto A = ops::Const(root, {{3.f, 5.f}, {-2.f, 0.f}});
  auto r = ops::Square(root.WithOpName("r"), A);

  std::vector<Tensor> outputs;
  ClientSession session(root);

  ASSERT_OK(session.Run({r}, &outputs));

  ASSERT_EQ(outputs[0].shape(), TensorShape({2, 2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(9.0, mat(0, 0));
  EXPECT_FLOAT_EQ(25.0, mat(0, 1));
  EXPECT_FLOAT_EQ(4.0, mat(1, 0));
  EXPECT_FLOAT_EQ(0.0, mat(1, 1));
}

TEST(tf_exec, Op_SquaredDifference) {
  Scope root = Scope::NewRootScope();
  Scope root_ngraph = root.NewSubScope("sub_scope_ngraph");
  root_ngraph = root_ngraph.WithDevice("/device:NGRAPH:0");

  auto A = ops::Placeholder(root, DataType::DT_FLOAT);
  auto B = ops::Placeholder(root, DataType::DT_FLOAT);
  auto r = ops::SquaredDifference(root_ngraph.WithOpName("r"), A, B);

  std::vector<Tensor> outputs;
  ClientSession session(root);

  ASSERT_OK(session.Run(
      {{A, {{3.f, 5.f}, {2.f, 0.f}}}, {B, {{1.f, 2.f}, {-1.f, 1.f}}}}, {r},
      &outputs));
  ASSERT_EQ(outputs[0].shape(), TensorShape({2, 2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(4.0, mat(0, 0));
  EXPECT_FLOAT_EQ(9.0, mat(0, 1));
  EXPECT_FLOAT_EQ(9.0, mat(1, 0));
  EXPECT_FLOAT_EQ(1.0, mat(1, 1));
}

TEST(tf_exec, Op_Rsqrt) {
  Scope root = Scope::NewRootScope();
  root = root.WithDevice("/device:NGRAPH:0");

  auto A = ops::Const(root, {{256.f, 16.f}, {4.f, 64.f}});
  auto r = ops::Rsqrt(root.WithOpName("r"), A);

  std::vector<Tensor> outputs;
  ClientSession session(root);

  ASSERT_OK(session.Run({r}, &outputs));

  ASSERT_EQ(outputs[0].shape(), TensorShape({2, 2}));

  auto mat = outputs[0].matrix<float>();
  EXPECT_FLOAT_EQ(1.f / 16.f, mat(0, 0));
  EXPECT_FLOAT_EQ(1.f / 4.f, mat(0, 1));
  EXPECT_FLOAT_EQ(1.f / 2.f, mat(1, 0));
  EXPECT_FLOAT_EQ(1.f / 8.f, mat(1, 1));
}

#undef ASSERT_OK

}  // namespace ngraph_bridge

}  // namespace tensorflow
