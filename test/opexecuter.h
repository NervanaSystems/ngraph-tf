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
#ifndef NGRAPH_TF_BRIDGE_OPEXECUTER_H_
#define NGRAPH_TF_BRIDGE_OPEXECUTER_H_

#include "ngraph/ngraph.hpp"
#include "ngraph_builder.h"
#include "ngraph_utils.h"
#include "test_utilities.h"
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

class OpExecuter {
 public:
  using NodeMetaData = map<Node*, vector<std::pair<Node*, int>>>;
  using NodeOutEdges = map<Node*, vector<const Edge*>>;

  OpExecuter(const Scope sc, const string test_op,
             const vector<DataType>& op_types,
             const std::vector<Output>& fetch_ops);

  ~OpExecuter();

  void ExecuteOnNGraph();
  void ExecuteOnTF();
  void CompareNgraphAndTF();

 private:
  Scope tf_scope_;
  const string test_op_type_;
  vector<Tensor> tf_inputs_;
  vector<Tensor> tf_outputs_;
  vector<Tensor> ngraph_outputs_;
  vector<Tensor> ngraph_outputs_;
  const vector<DataType> expected_output_datatypes_;
  const vector<const Tensor*>& static_input_map_;
  
  // To Do : For placeholder const FeedType sess_run_inputs_;
  const std::vector<Output> sess_run_fetchoutputs_;

  void GetNodeData(Graph& graph, NodeMetaData& node_inedge_md,
                   NodeMetaData& node_outedge_md, NodeOutEdges& node_outedges,
                   Node** test_op);
  void ValidateGraph(const Graph& graph, const vector<string> allowed_nodes);
  void CreateNodeDef(const string op_type, const string op_name_prefix,
                     int index, const DataType dt, NodeDef& node_def);
};

}  // namespace testing
}  // namespace ngraph_bridge

}  // namespace tensorflow

#endif  // NGRAPH_TF_BRIDGE_OPEXECUTER_H_
