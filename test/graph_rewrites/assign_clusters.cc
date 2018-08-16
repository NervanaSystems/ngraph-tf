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

#include "ngraph_assign_clusters.h"
#include "ngraph_utils.h"

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

#define ASSERT_OK(x) ASSERT_EQ((x), ::tensorflow::Status::OK());

// Given a graph of this form:
//
//  Node1--->Node2
//    \       /
//     \     /
//      |   |
//      v*  v
//      Node3
//
// where the starred input is static, we want to make sure that Node2 and Node3
// are not accidentally coalesced by the following chain of events:
//
// Node1-->Node2 coalesced
// Node2-->Node3 coalesced   **actually invalid, because Node1 is now in same
//                             cluster as Node2.
TEST(assign_clusters, cone) {
  Graph g(OpRegistry::Global());

  Tensor t(DT_FLOAT, TensorShape{2,3});

  Node* node1;
  ASSERT_OK(NodeBuilder("node1","Const")
              .Attr("dtype",DT_FLOAT)
              .Attr("value",t)
              .Attr("_ngraph_marked_for_clustering",true)
              .Finalize(&g,&node1));

  Node* node2;
  ASSERT_OK(NodeBuilder("node2","Shape")
              .Input(node1, 0)
              .Attr("T",DT_FLOAT)
              .Attr("out_type",DT_INT32)
              .Attr("_ngraph_marked_for_clustering",true)
              .Finalize(&g,&node2));

  Node* node3;
  ASSERT_OK(NodeBuilder("node3","Reshape")
              .Input(node1, 0)
              .Input(node2, 0)
              .Attr("T",DT_FLOAT)
              .Attr("Tshape",DT_INT32)
              .Attr("_ngraph_marked_for_clustering",true)
              .Attr("_ngraph_static_inputs",std::vector<int32>{1})
              .Finalize(&g,&node3));

  ASSERT_OK(AssignClusters(&g));
  int node2_cluster, node3_cluster;
  ASSERT_OK(GetNodeCluster(node2,&node2_cluster));
  ASSERT_OK(GetNodeCluster(node3,&node3_cluster));
  ASSERT_NE(node2_cluster,node3_cluster);
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
