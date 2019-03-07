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

#include "ngraph_optimizer.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
//#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
//#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"


#include <iomanip>

#include <iostream>

using namespace std;

namespace tensorflow {
namespace ngraph_bridge {

Status NgraphOptimizer::Optimize(tensorflow::grappler::Cluster* cluster, const tensorflow::grappler::GrapplerItem& item,
                                 GraphDef* output) {

  VLOG(1) << "NGRAPH-TF OPTIMIZER called ";
  *output = item.graph;
  //GraphDef optimized_graph_def;

  /*GraphConstructorOptions opts;
  Graph graph_(OpRegistry::Global());
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, item.graph, &graph_));

    // For filename generation purposes, grab a fresh index. This is just an
    // arbitrary integer to avoid filename collisions resulting from subsequent
    // runs of this pass.
    int idx = FreshIndex();

    // If requested, dump pre-capture graphs.
    if (DumpPrecaptureGraphs()) {
      DumpGraphs(graph_, idx, "precapture", "Pre-Capture Graph");
    }

    // If ngraph is disabled via ngraph_bridge api or NGRAPH_TF_DISABLE is set
    // we will not do anything; all subsequent
    // passes become a no-op.
    if (config::IsEnabled() == false ||
        std::getenv("NGRAPH_TF_DISABLE") != nullptr) {
      return Status::OK();
    }

    // Do variable capture then, if requested, dump the graphs.
    TF_RETURN_IF_ERROR(CaptureVariables(&graph_));
    if (DumpCapturedGraphs()) {
      DumpGraphs(graph_, idx, "captured", "Graph With Variables Captured");
    }*/
  
  //graph_.ToGraphDef(output);
  return Status::OK();
}

void NgraphOptimizer::Feedback(tensorflow::grappler::Cluster* cluster, const tensorflow::grappler::GrapplerItem& item,
                               const GraphDef& optimize_output, double result) {
  // no-op
}

void NgraphOptimizer::DumpGraphs(Graph &graph, int idx,
                  std::string filename_prefix, std::string title) {
    // If we have a "main" graph, dump that.
      auto dot_filename = DotFilename(filename_prefix, idx);
      auto pbtxt_filename = PbtxtFilename(filename_prefix, idx);
      NGRAPH_VLOG(0) << "NGRAPH-TF OPTIMIZER Dumping main graph to " << dot_filename;
      NGRAPH_VLOG(0) << "NGRAPH-TF OPTIMIZER Dumping main graph to " << pbtxt_filename;

      GraphToDotFile(&graph, dot_filename, title);
      GraphToPbTextFile(&graph, pbtxt_filename);
  }

REGISTER_GRAPH_OPTIMIZER_AS(NgraphOptimizer, "ng-optimizer");

}  // end namespace ngraph_bridge

}  // end namespace tensorflow
