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

/*******************************************************************************

This test is inspired from the deadness test, mentioned in the commit message of
the deadness analysis found in the below revision

Github repository: https://github.com/tensorflow/tensorflow
Revision: 6619dd5fdcad02f087f5758083e2585bdfef9e78

Quoted from the commit message **
TensorFlow allows nodes to have some live inputs and some dead inputs.  The
executor does not execute these nodes but instead propagates a dead signal to
all their outputs (i.e. these nodes are treated as fully dead).

This is a problem for auto-clustering because it means auto-clustering can kill
nodes that used to be alive.  For instance say before clustering we have a graph
like

digraph {
  Alive0 -> P
  Alive1 -> Q
  Dead -> R
  P -> X
  Q -> X
  Q -> Y
  R -> Y
}

and we cluster P, Q, R, X and Y into a single XLA cluster.

Then after clustering both X and Y are dead because the cluster is a single node
as far as the executor is concerned and said node won't get scheduled if any of
its inputs are dead.

*******************************************************************************/

#include "../test_utilities.h"
#include "gtest/gtest.h"
#include "ngraph_assign_clusters.h"
#include "ngraph_mark_for_clustering.h"
#include "ngraph_utils.h"
#include "tf_graph_writer.h"

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing {

#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());

TEST(DeadnessCheck, livedead1NGRAPH) {
  Scope root = Scope::NewRootScope();

  auto A = ops::Placeholder(root, DataType::DT_FLOAT);
  auto B = ops::Placeholder(root, DataType::DT_FLOAT);
  auto C = ops::Placeholder(root, DataType::DT_FLOAT);
  auto pred = ops::Placeholder(root, DataType::DT_BOOL);

  auto S = ops::Switch(root, A, pred);
  auto P = ops::Add(root, A, B);

  auto Q = ops::Add(root, A, C);
  auto R = ops::Sub(root, S.output_true, B);

  auto M = ops::Mul(root, P, Q);
  auto D = ops::RealDiv(root, Q, R);

  std::vector<Tensor> outputs;
  ClientSession session(root);

  ASSERT_NE(
      session.Run(
          {{A, {3.f, 5.f}}, {B, {3.f, 2.f}}, {C, {3.f, 2.f}}, {pred, false}},
          {M, D}, &outputs),
      Status::OK());
}

TEST(DeadnessCheck, livedead1TF) {
  Scope root = Scope::NewRootScope();
  DeactivateNGraph();

  auto A = ops::Placeholder(root, DataType::DT_FLOAT);
  auto B = ops::Placeholder(root, DataType::DT_FLOAT);
  auto C = ops::Placeholder(root, DataType::DT_FLOAT);
  auto pred = ops::Placeholder(root, DataType::DT_BOOL);

  auto S = ops::Switch(root, A, pred);
  auto P = ops::Add(root, A, B);

  auto Q = ops::Add(root, A, C);
  auto R = ops::Sub(root, S.output_true, B);

  auto M = ops::Mul(root, P, Q);
  auto D = ops::RealDiv(root, Q, R);

  std::vector<Tensor> outputs;
  ClientSession session(root);

  ASSERT_NE(
      session.Run(
          {{A, {3.f, 5.f}}, {B, {3.f, 2.f}}, {C, {3.f, 2.f}}, {pred, false}},
          {M, D}, &outputs),
      Status::OK());
}

TEST(DeadnessCheck, DTestPl) {
  Scope root = Scope::NewRootScope();

  auto dataX = ops::Placeholder(root.WithOpName("dataX"), DataType::DT_FLOAT);
  auto dataY = ops::Placeholder(root.WithOpName("dataY"), DataType::DT_FLOAT);
  auto dataZ = ops::Placeholder(root.WithOpName("dataZ"), DataType::DT_FLOAT);
  auto A = ops::Placeholder(root.WithOpName("A"), DataType::DT_FLOAT);
  auto B = ops::Const(root.WithOpName("B"), {3.f, 2.f});

  auto predX = ops::Placeholder(root.WithOpName("PredX"), DataType::DT_BOOL);
  auto predY = ops::Placeholder(root.WithOpName("PredY"), DataType::DT_BOOL);
  auto predZ = ops::Placeholder(root.WithOpName("PredZ"), DataType::DT_BOOL);

  auto SX = ops::Switch(root.WithOpName("SwitchX"), dataX, predX);
  auto SY = ops::Switch(root.WithOpName("SwitchY"), dataY, predY);
  auto SZ = ops::Switch(root.WithOpName("SwitchZ"), dataZ, predZ);

  auto XYAdd =
      ops::Add(root.WithOpName("XYAdd"), SX.output_true, SY.output_false);
  auto XYMul = ops::Mul(root.WithOpName("XYMul"), XYAdd, B);
  auto XYZSub = ops::Sub(root.WithOpName("XYZSub"), SZ.output_true, XYMul);
  auto XYZMul = ops::Mul(root.WithOpName("XYZMul"), XYZSub, B);
  auto YAdd = ops::Add(root.WithOpName("YAdd"), SY.output_true, A);
  auto YMul = ops::Mul(root.WithOpName("YMul"), YAdd, B);

  // Graph graph(OpRegistry::Global());
  // TF_CHECK_OK(root.ToGraph(&graph));

  // // for (const Edge* e : graph.edges()) {
  // //   NGRAPH_VLOG(5) << "Edge between, Src: " << e->src()->name()
  // //                  << " Src op index " << e->src_output()
  // //                  << " ,Dst: " << e->dst()->name() << " dst ip index "
  // //                  << e->dst_input();
  // // }

  // ASSERT_OK(MarkForClustering(&graph));
  // ASSERT_OK(AssignClusters(&graph));

  // int XYAdd_cluster, XYMul_cluster, XYZSub_cluster, XYZMul_cluster,
  //     YAdd_cluster, YMul_cluster;
  // ASSERT_OK(GetNodeCluster(XYAdd.node(), &XYAdd_cluster));
  // ASSERT_OK(GetNodeCluster(XYMul.node(), &XYMul_cluster));
  // ASSERT_OK(GetNodeCluster(XYZSub.node(), &XYZSub_cluster));
  // ASSERT_OK(GetNodeCluster(XYZMul.node(), &XYZMul_cluster));
  // ASSERT_OK(GetNodeCluster(YAdd.node(), &YAdd_cluster));
  // ASSERT_OK(GetNodeCluster(YMul.node(), &YMul_cluster));

  // ASSERT_EQ(XYAdd_cluster, XYMul_cluster);
  // ASSERT_EQ(XYZSub_cluster, XYZMul_cluster);
  // ASSERT_EQ(YAdd_cluster, YMul_cluster);

  std::vector<Tensor> outputs;
  ClientSession session(root);
  ASSERT_NE(session.Run({{dataX, {3.f, 5.f}},
                         {dataY, {3.f, 2.f}},
                         {dataZ, {3.f, 2.f}},
                         {A, {3.f, 2.f}},
                         //{B, {3.f, 2.f}},
                         {predX, true},
                         {predY, false},
                         {predZ, true}},
                        {XYZMul, YMul}, &outputs),
            Status::OK());
}

}  // namespace testing
}  // namespace ngraph_bridge
}  // namespace tensorflow
