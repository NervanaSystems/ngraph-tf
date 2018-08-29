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

#include "ngraph_builder1.h"
#include "ngraph_translateops.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

Status Builder1::TranslateGraph(
    const std::vector<TensorShape>& inputs,
    const std::vector<const Tensor*>& static_input_map,
    shared_ptr<ng::Function>& ng_function) {
  TF_RETURN_IF_ERROR(Initialize());

  vector<shared_ptr<ng::op::Parameter>> ng_parameter_list;
  TF_RETURN_IF_ERROR(GetInputParams(inputs, tf_params, ng_parameter_list));

  TF_RETURN_IF_ERROR(TranslateEachOp(tf_ops, static_input_map));

  VectNg ng_result_list;
  TF_RETURN_IF_ERROR(GetOutputNodes(tf_ret_vals, ng_result_list));

  // Create the nGraph function.
  try {
    ng_function = make_shared<ng::Function>(ng_result_list, ng_parameter_list);
  } catch (...) {
    return errors::Unimplemented("Unable to create nGraph function");
  }

  //
  // Request row-major layout on results.
  //
  for (auto result : ng_function->get_results()) {
    result->set_needs_default_layout(true);
  }
  return Status::OK();
}

Status Builder1::TranslateGraph(
    OpKernelContext* ctx, std::shared_ptr<ngraph::Function>& ng_function) {
  TF_RETURN_IF_ERROR(Initialize());

  std::vector<const Tensor*> static_input_map;

  std::vector<TensorShape> inputs(ctx->num_inputs());

  static_input_map.resize(ctx->num_inputs());
  for (int i = 0; i < ctx->num_inputs(); i++) {
    const Tensor& input_tensor = ctx->input(i);
    if (m_input_is_static[i]) {
      static_input_map[i] = &input_tensor;
    }
    inputs[i] = input_tensor.shape();
  }

  return TranslateGraph(inputs, static_input_map, ng_function);
}

Status Builder1::TranslateEachOp(
    const vector<const Node*>& tf_ops,
    const std::vector<const Tensor*>& static_input_map) {
  // Create the nGraph ops from TensorFlow ops.
  for (auto op : tf_ops) {
    NGRAPH_VLOG(2) << "Constructing op " << op->name() << " which is "
                   << op->type_string();

    Builder1::TranslatorFn translate_fn;
    vector<int> input_indexes;
    TF_RETURN_IF_ERROR(
        GetOpTranslationRequirements(op, translate_fn, input_indexes));
    // input_indexes can be size 0 (to indicate/handle variadic inputs nodes
    // like Addn)
    VectNg subgraph_out_nodes(op->num_outputs());

    bool variadic_input = input_indexes.size() == 0;
    int num_inputs = variadic_input ? op->num_inputs() : input_indexes.size();
    VectNg ng_arg_vec(num_inputs);
    if (op->type_string() != "Const") {
      for (int idx = 0; idx < num_inputs; idx++) {
        TF_RETURN_IF_ERROR(GetInputNode(
            op, (variadic_input ? idx : input_indexes[idx]), &ng_arg_vec[idx]));
      }
    }
    TF_RETURN_IF_ERROR(
        translate_fn(op, ng_arg_vec, static_input_map, subgraph_out_nodes));

    ng_op_map[op->name()] = subgraph_out_nodes;
  }
  return Status::OK();
}

Status Builder1::ClassifyNodes(const vector<Node*>& ordered) {
  // Split ops into params, retvals, and all others.
  for (const auto n : ordered) {
    if (n->IsSink() || n->IsSource()) {
      continue;
    }

    if (n->IsControlFlow()) {
      return errors::Unimplemented(
          "Encountered a control flow op in the nGraph bridge: ",
          n->DebugString());
    }
    if (n->type_string() == "_Arg") {
      tf_params.push_back(n);
    } else if (n->type_string() == "_Retval") {
      tf_ret_vals.push_back(n);
    } else {
      tf_ops.push_back(n);
    }
  }
  return Status::OK();
}

Status Builder1::GetInputParams(
    const std::vector<TensorShape>& inputs, vector<const Node*> tf_params,
    vector<shared_ptr<ng::op::Parameter>>& ng_parameter_list) {
  // Populate the parameter list, and also put parameters into the op map.

  ng_parameter_list.resize(tf_params.size());

  for (auto parm : tf_params) {
    DataType dtype;
    if (GetNodeAttr(parm->attrs(), "T", &dtype) != Status::OK()) {
      return errors::InvalidArgument("No data type defined for _Arg");
    }
    int index;
    if (GetNodeAttr(parm->attrs(), "index", &index) != Status::OK()) {
      return errors::InvalidArgument("No index defined for _Arg");
    }

    ng::element::Type ng_et;
    TF_RETURN_IF_ERROR(TFDataTypeToNGraphElementType(dtype, &ng_et));

    ng::Shape ng_shape;
    TF_RETURN_IF_ERROR(TFTensorShapeToNGraphShape(inputs[index], &ng_shape));

    auto ng_param = make_shared<ng::op::Parameter>(ng_et, ng_shape);
    ng_op_map[parm->name()].push_back(ng_param);
    ng_parameter_list[index] = ng_param;
  }
  return Status::OK();
}

Status Builder1::GetOutputNodes(const vector<const Node*>& tf_ret_vals,
                                VectNg& ng_result_list) {
  // Populate the result list.

  ng_result_list.resize(tf_ret_vals.size());

  for (auto n : tf_ret_vals) {
    // Make sure that this _Retval only has one input node.
    if (n->num_inputs() != 1) {
      return errors::InvalidArgument("_Retval has ", n->num_inputs(),
                                     " inputs, should have 1");
    }

    int index;
    if (GetNodeAttr(n->attrs(), "index", &index) != Status::OK()) {
      return errors::InvalidArgument("No index defined for _Retval");
    }

    shared_ptr<ng::Node> result;
    TF_RETURN_IF_ERROR(GetInputNode(n, 0, &result));

    ng_result_list[index] = result;
  }
  return Status::OK();
}

Status Builder1::Initialize() {
  if (!is_initialized) {
    //
    // We will visit ops in topological order.
    //
    // ought to be `const Node*`, but GetReversePostOrder doesn't use `const`

    vector<Node*> ordered;
    GetReversePostOrder(tf_graph, &ordered);

    TF_RETURN_IF_ERROR(ClassifyNodes(ordered));
    //
    // Initialize the "m_input_is_static" vector as follows:
    // (1) create m_input_is_static with n+1 elements, where n is the max arg
    //     index
    // (2) for each _Arg node n, set m_input_is_static[n.index] to true if n
    //     is driving any static input; else set it to false.
    //

    // Create the vector.
    int32 max_arg_index = -1;
    std::vector<const Node*> arg_nodes;

    for (auto node : tf_graph.nodes()) {
      if (node->type_string() == "_Arg") {
        arg_nodes.push_back(node);
        int32 index;
        TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &index));
        if (index > max_arg_index) max_arg_index = index;
      }
    }

    m_input_is_static = std::vector<bool>(max_arg_index + 1, false);

    // Fill the vector.
    for (auto node : arg_nodes) {
      int32 index;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &index));

      for (auto edge : node->out_edges()) {
        if (edge->IsControlEdge() || !edge->dst()->IsOp()) {
          continue;
        }

        NGRAPH_VLOG(5) << "For arg " << index << " checking edge "
                       << edge->DebugString();

        if (InputIsStatic(edge->dst(), edge->dst_input())) {
          NGRAPH_VLOG(5) << "Marking edge static: " << edge->DebugString();
          m_input_is_static[index] = true;
          break;
        }
      }
      NGRAPH_VLOG(5) << "Marking arg " << index
                     << " is static: " << m_input_is_static[index];
    }
    is_initialized = true;
  }
  return Status::OK();
}

Status Builder1::GetInputNode(const Node* op, size_t input_idx,
                              shared_ptr<ng::Node>* result) {
  // input op may have resulted in more than one ng::Node (eg. Split)
  // we need to look at Edge to check index of the input op
  std::vector<const Edge*> edges;
  TF_RETURN_IF_ERROR(op->input_edges(&edges));
  size_t src_output_idx;
  try {
    src_output_idx = edges.at(input_idx)->src_output();
  } catch (const out_of_range&) {
    return Status(tensorflow::error::NOT_FOUND, "Edge not found");
  }
  Node* tf_input;
  TF_RETURN_IF_ERROR(op->input_node(input_idx, &tf_input));
  try {
    *result = ng_op_map.at(tf_input->name()).at(src_output_idx);
  } catch (const out_of_range&) {
    return Status(tensorflow::error::NOT_FOUND, "Input node not found");
  }
  return Status::OK();
}

// Helper function to translate a binary op in cases where there is a one-to-one
// mapping from TensorFlow ops to nGraph ops.
//
// Example usage:
//
//  if (n->type_string == "Add") {
//    TF_RETURN_IF_ERROR(TranslateBinary<ng::op::Add>(op, ng_arg_vec,
//    static_input_map,
//    subgraph_out_nodes));
//  }
//
template <typename T>
Status TranslateBinary(const Node* op, VectNg& ng_arg_vec,
                       const std::vector<const Tensor*>& static_input_map,
                       VectNg& subgraph_out_nodes) {
  if (subgraph_out_nodes.size() != 1)
    return errors::InvalidArgument(
        "TranslateBinary call for ", op->type_string(), " expects ",
        subgraph_out_nodes.size(), " outputs. Should have been 1");
  if (ng_arg_vec.size() != 2)
    return errors::InvalidArgument("TranslateBinary call for ",
                                   op->type_string(), " expects 2 inputs. Got ",
                                   ng_arg_vec.size(), ".");
  auto node_pair = ng::builder::numpy_broadcast(
      std::make_pair(ng_arg_vec[0], ng_arg_vec[1]));
  subgraph_out_nodes[0] = make_shared<T>(node_pair.first, node_pair.second);
  return Status::OK();
}

template <typename T>
Status TranslateUnary(const Node* op, const VectNg& ng_arg_vec,
                      const std::vector<const Tensor*>& static_input_map,
                      VectNg& subgraph_out_nodes) {
  if (subgraph_out_nodes.size() != 1)
    return errors::InvalidArgument(
        "TranslateUnary call for ", op->type_string(), " expects ",
        subgraph_out_nodes.size(), " outputs. Should have been 1");
  if (ng_arg_vec.size() != 1)
    return errors::InvalidArgument("TranslateUnary call for ",
                                   op->type_string(), " expects 1 inputs. Got ",
                                   ng_arg_vec.size(), ".");

  subgraph_out_nodes[0] = make_shared<T>(ng_arg_vec[0]);
  return Status::OK();
}

const std::map<const string, Builder1::TranslatorFn> Builder1::TRANSLATE_OP_MAP{
    {"Abs", TranslateUnary<ngraph::op::Abs>},
    {"Add", TranslateBinary<ngraph::op::Add>},
    {"AddN", TranslateAddNOp},
    {"AvgPool", TranslateAvgPoolOp},
    {"Const", TranslateConstOp},
    {"Equal", TranslateBinary<ngraph::op::Equal>},
    {"Exp", TranslateUnary<ngraph::op::Exp>},
    {"Floor", TranslateUnary<ngraph::op::Floor>},
    {"FloorDiv", TranslateFloorDivOp},
    {"FloorMod", TranslateFloorModOp},
    {"Greater", TranslateBinary<ngraph::op::Greater>},
    {"GreaterEqual", TranslateBinary<ngraph::op::GreaterEq>},
    {"Less", TranslateBinary<ngraph::op::Less>},
    {"LessEqual", TranslateBinary<ngraph::op::LessEq>},
    {"Log", TranslateUnary<ngraph::op::Log>},
    {"LogicalAnd", TranslateBinary<ngraph::op::And>},
    {"LogicalNot", TranslateUnary<ngraph::op::Not>},
    {"Maximum", TranslateBinary<ngraph::op::Maximum>},
    {"Minimum", TranslateBinary<ngraph::op::Minimum>},
    {"Mul", TranslateBinary<ngraph::op::Multiply>},
    {"Neg", TranslateUnary<ngraph::op::Negative>},
    {"Neg", TranslateUnary<ngraph::op::Negative>},
    {"NoOp", [](const Node* op, VectNg& ng_arg_vec,
                const std::vector<const Tensor*>& static_input_map,
                VectNg& subgraph_out_nodes) { return Status::OK(); }},
    {"Pow", TranslateBinary<ngraph::op::Power>},
    {"RealDiv", TranslateBinary<ngraph::op::Divide>},
    {"Sign", TranslateUnary<ngraph::op::Sign>},
    {"Sqrt", TranslateUnary<ngraph::op::Sqrt>},
    {"Sub", TranslateBinary<ngraph::op::Subtract>},
    {"Tanh", TranslateUnary<ngraph::op::Tanh>}};

const std::map<const string, vector<int>> Builder1::INPUT_INDEX_MAP{};

Status Builder1::GetOpTranslationRequirements(
    const Node* op, Builder1::TranslatorFn& translate_fn,
    vector<int>& input_indexes) {
  // This function wraps TRANSLATE_OP_MAP.
  // It returns a translate function and input indexes
  // The translate function MUST be present in TRANSLATE_OP_MAP
  // input_idx may not be present, since it can be inferred from op
  auto iter_fn = TRANSLATE_OP_MAP.find(op->type_string());
  if (iter_fn != TRANSLATE_OP_MAP.end()) {
    translate_fn = iter_fn->second;
  } else {
    // -----------------------------
    // Catch-all for unsupported ops
    // -----------------------------
    NGRAPH_VLOG(3) << "Unsupported Op: " << op->name() << " ("
                   << op->type_string() << ")";
    NGRAPH_VLOG(3) << op->def().DebugString();
    return errors::InvalidArgument("Unsupported Op: ", op->name(), " (",
                                   op->type_string(), ")");
  }

  auto iter_input_indexes = INPUT_INDEX_MAP.find(op->type_string());
  if (iter_input_indexes != INPUT_INDEX_MAP.end()) {
    input_indexes = iter_input_indexes->second;
  } else {
    // By default we should return {0,1, ..., (op->num_inputs)-1}...unless
    // otherwise specified.
    input_indexes.resize(op->num_inputs());
    std::iota(std::begin(input_indexes), std::end(input_indexes),
              0);  // iota: Populate with increasing integers 1,2...
  }
  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
