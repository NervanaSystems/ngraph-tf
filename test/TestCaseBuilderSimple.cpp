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
#include "TestCaseBuilderSimple.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

namespace testing{

void BuilderTestSimple::CreateNodeDef(const string op_type, const string op_name_prefix,int index, const DataType dt, NodeDef& node_def){
  string new_node_name = op_name_prefix + std::to_string(index);
  node_def.set_name(new_node_name);
  node_def.set_op(op_type);
  SetAttrValue(dt,
                &((*(node_def.mutable_attr()))["T"]));
  SetAttrValue(index, &((*(node_def.mutable_attr()))["index"]));
}

void BuilderTestSimple::GetNodeData(Graph& graph, NodeMetaData& node_inedge_md,
                                    NodeMetaData& node_outedge_md,
                                    NodeOutEdges& node_outedges,
                                    Node** test_op) {
  bool found_test_op = false;
  for (const Edge* e : graph.edges()) {
    if(!found_test_op){
      if(e->src()->IsOp() && (e->src()->type_string())==test_op_type_){
        found_test_op = true;
        *test_op=e->src();
      }
      if(e->dst()->IsOp() && (e->dst()->type_string())==test_op_type_){
        found_test_op = true;
        *test_op=e->dst();
      }
    }
    NGRAPH_VLOG(5) << "Edge between, Src: " << e->src()->name()
                   << " Src op index " << e->src_output()
                   << " ,Dst: " << e->dst()->name() << " dst ip index "
                   << e->dst_input();
    // update src's outedge metadata
    node_outedge_md[e->src()].push_back({e->dst(), e->dst_input()});
    node_inedge_md[e->dst()].push_back({e->src(), e->src_output()});
    node_outedges[e->src()].push_back(e);
  }
}

// TO_DO check for vector allowed_nodes
// when we allow other than "Const" node type as input
// Validate that the graph has n allowed_nodes and 1 test_op_type node
void BuilderTestSimple::ValidateGraph(const Graph& graph,
                                      const vector<string> allowed_nodes) {
  NGRAPH_VLOG(5) << "Validate graph";
  bool found_test_op = false;
  for (Node* node : graph.nodes()) {
    if (node->IsSource() || node->IsSink()) {
      continue;
    } else if (node->type_string() == test_op_type_) {
      // only one node of type test_op
      ASSERT_FALSE(found_test_op);
      found_test_op = true;
    } else {
      ASSERT_TRUE(node->type_string() == allowed_nodes[0])
          << "Found Not allowed Op: " << node->type_string();
    }
  }
  NGRAPH_VLOG(5) << "Validate graph done";
}

BuilderTestSimple::BuilderTestSimple(const Scope sc, const string test_op,
                                     const vector<DataType>& op_types, const std::vector<Output>& sess_run_fetchops)
    : tf_scope_(sc),
      test_op_type_(test_op),
      expected_output_datatypes_(op_types),
      sess_run_fetchoutputs_(sess_run_fetchops) {
  NGRAPH_VLOG(5) << " test op " << test_op;
  for (auto dt : expected_output_datatypes_) {
    NGRAPH_VLOG(5) << dt;
  }
}

BuilderTestSimple::~BuilderTestSimple(){}

void BuilderTestSimple::ExecuteOnTF(){
  DummyDeactivateNGraph();
  ClientSession session(tf_scope_);
  ASSERT_EQ(Status::OK(),session.Run(sess_run_fetchoutputs_,&tf_outputs_));
}

void BuilderTestSimple::CompareNgraphAndTF(){
  ASSERT_EQ(tf_outputs_.size(), ngraph_outputs_.size());
  for(int i=0; i<tf_outputs_.size();i++){
    DummyAssertTensorEquals(tf_outputs_[i], ngraph_outputs_[i]);
  }
}

void BuilderTestSimple::ExecuteOnNGraph() {
  Graph graph(OpRegistry::Global());
  TF_CHECK_OK(tf_scope_.ToGraph(&graph));

  // For debug
  GraphToPbTextFile(&graph, "tf_graph.pbtxt");

  ValidateGraph(graph, {"Const"});
  
  NodeMetaData node_inedge_metadata;
  NodeMetaData node_outedge_metadata;
  NodeOutEdges node_out_edges;
  Node* test_op;

  GetNodeData(graph, node_inedge_metadata, node_outedge_metadata,
              node_out_edges, &test_op);
  NGRAPH_VLOG(5) << "Got graph data. Found op "<< test_op->type_string();

  // Get Tensor input shapes and values from the const nodes
  int number_of_inputs = test_op->num_inputs();
  vector<TensorShape> input_shapes;
  vector<Node*> input_node;
  
  for (int i = 0; i < number_of_inputs; i++) {
    Node* ip;
    Tensor ip_tensor;
    ASSERT_EQ(Status::OK(), test_op->input_node(i, &ip));
    input_node.push_back(ip);
    ASSERT_EQ(Status::OK(), GetNodeAttr(ip->attrs(), "value", &ip_tensor));
    input_shapes.push_back(ip_tensor.shape());
    tf_inputs_.push_back(ip_tensor);
    NGRAPH_VLOG(5) << " Extracted tensor from const " << i << " "
                   << tf_inputs_[i].DebugString();
  }

  NGRAPH_VLOG(5) << "Got input nodes and tensors";

  // Replace the input nodes to Test_op with _Arg nodes
  for (int i = 0; i < number_of_inputs; i++) {
    Node* ip_node = input_node[i];
    NodeDef new_arg_node_def;
    CreateNodeDef("_Arg","arg_",i,tf_inputs_[i].dtype(),new_arg_node_def);
    
    // Add node to graph
    Status status;
    Node* arg_node = graph.AddNode(new_arg_node_def, &status);
    ASSERT_EQ(Status::OK(), status);

    // Remove the Const Node
    graph.RemoveNode(input_node[i]);

    // Add edge from SOURCE to _Arg
    auto src_nodes_metadata = node_inedge_metadata[ip_node];
    for (int j = 0; j < src_nodes_metadata.size(); j++) {
      graph.AddEdge(src_nodes_metadata[j].first, src_nodes_metadata[j].second,
                    arg_node, 0);
    }
    // Adds an edge from arg_node to test_op
    graph.AddEdge(arg_node, 0, test_op, i);
  }

  NGRAPH_VLOG(5) << "Replaced input nodes with _Arg";

  // Add _Retval to graph
  int number_of_outputs = expected_output_datatypes_.size();
  // For all the output edges from test_op (there should be only one, to SINK)
  // get the dest node and the
  // destination_input_index
  // (TO DO : ) ADD ASSERT to check one?
  auto dest_nodes_metadata = node_outedge_metadata[test_op];

  // Remove edges from test_op to SINK (not removing might be also ok)
  for (const Edge* e : node_out_edges[test_op]) {
    graph.RemoveEdge(e);
  }

  for (int i = 0; i < number_of_outputs; i++) {
    // Add new retval_ node
    NodeDef new_ret_node_def;
    CreateNodeDef("_Retval","retval_",i,expected_output_datatypes_[i],new_ret_node_def);
    Status status;
    Node* ret_node = graph.AddNode(new_ret_node_def, &status);
    ASSERT_EQ(Status::OK(), status);

    // Add edges from _Retval to sink
    for (int j = 0; j < dest_nodes_metadata.size(); j++) {
      graph.AddEdge(ret_node, 0, dest_nodes_metadata[j].first,
                    dest_nodes_metadata[j].second);
    }
    // Add edges from test_op to _Retval
    graph.AddEdge(test_op, i, ret_node, 0);
  }

  NGRAPH_VLOG(5) << "Added _Retval nodes ";

  NGRAPH_VLOG(5) << "After rewrite *** ";
  for (const Edge* e : graph.edges()) {
    NGRAPH_VLOG(5) << "Edge between, Src: " << e->src()->name()
                   << " ,Dst: " << e->dst()->name();
  }
  // For debug
  GraphToPbTextFile(&graph, "rewrite_ngraph.pbtxt");

  // Create nGraph function
  NGRAPH_VLOG(5) << " Create ng function ";
  shared_ptr<ng::Function> ng_function;
  ASSERT_EQ(Status::OK(),
            Builder::TranslateGraph(input_shapes, &graph, ng_function));
  
  // ng function should get same number of outputs
  ASSERT_EQ(expected_output_datatypes_.size(), ng_function->get_output_size());

  // Create nGraph backend
  // Create the nGraph backend
  auto backend = ng::runtime::Backend::create("CPU");

  // Allocate tensors for inputs
  vector<std::shared_ptr<ngraph::runtime::TensorView>> ng_ip_tensors;
  vector<std::shared_ptr<ngraph::runtime::TensorView>> ng_op_tensors;
  
  NGRAPH_VLOG(5) << " Creating ng inputs ";
  for (int i = 0; i < number_of_inputs; i++) {
    ng::Shape ng_shape;
    ASSERT_EQ(Status::OK(),
              TFTensorShapeToNGraphShape(tf_inputs_[i].shape(), &ng_shape));
    ng::element::Type ng_et;
    ASSERT_EQ(Status::OK(),
              TFDataTypeToNGraphElementType(tf_inputs_[i].dtype(), &ng_et));
    void* src_ptr = (void*)DMAHelper::base(&tf_inputs_[i]);
    auto result = backend->create_tensor(ng_et, ng_shape, src_ptr);
    ng_ip_tensors.push_back(result);
  }

  NGRAPH_VLOG(5) << " Creating ng outputs ";
  vector<TensorShape> tf_op_shapes;
  for (int i = 0; i < number_of_outputs; i++) {
    auto ng_op_shape = ng_function->get_output_shape(i);
    auto ng_op_type = ng_function->get_output_element_type(i);

    ng::element::Type ng_et_expected;
    ASSERT_EQ(Status::OK(), TFDataTypeToNGraphElementType(expected_output_datatypes_[i],
                                                          &ng_et_expected));

    // Expected element type should match ng_op_type
    ASSERT_EQ(ng_et_expected, ng_op_type);
    vector<int64> dims;
    for (auto dim : ng_op_shape) {
      dims.push_back(dim);
    }
    TensorShape tf_shape(dims);
    tf_op_shapes.push_back(tf_shape);
    auto result = backend->create_tensor(ng_op_type, ng_op_shape);
    ng_op_tensors.push_back(result);
  }

  // Execute the nGraph
  NGRAPH_VLOG(5) << " Executing on nGraph ";
  backend->call(ng_function, ng_op_tensors, ng_ip_tensors);
  NGRAPH_VLOG(5) << " Writing to Tensors ";
  for (auto i = 0; i < ng_function->get_output_size(); i++) {
    // Convert to tf tensor
    Tensor output_tensor(expected_output_datatypes_[i], tf_op_shapes[i]);
    void* dst_ptr = DMAHelper::base(&output_tensor);
    ng_op_tensors[i]->read(dst_ptr, 0, output_tensor.TotalBytes());
    ngraph_outputs_.push_back(output_tensor);
    //DumpNGTensor(cout, ng_function->get_output_op(i)->get_name(), ng_op_tensors[i]);
  }

  } // ExecuteOnNGraph

} //namespace testing
}  // namespace ngraph_bridge

}  // namespace tensorflow
