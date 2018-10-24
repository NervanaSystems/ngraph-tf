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
#include "ngraph_disable_assert.h"

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/types.h"

#include "ngraph_utils.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

//
// Main entry point for disable assert.
//
Status DisableAssert(Graph* graph) {
  std::vector<const Edge*> edges;
  for (auto node : graph->op_nodes()) {
    if (node->type_string() == "Assert") {
      NGRAPH_VLOG(4) << "Checking: " << node->name();
      for (auto edge : node->out_edges()) {
        if (edge->IsControlEdge()) {
          if (edge != NULL) {
            NGRAPH_VLOG(4) << "Collecting all the control edges";
            edges.push_back(edge);
          }
        }
      }
    }
  }
  for (auto edge : edges) {
    NGRAPH_VLOG(4) << "Removing control edge: " << edge->DebugString();
    graph->RemoveControlEdge(edge);
  }
  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
