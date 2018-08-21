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
#ifndef NGRAPH_TF_BRIDGE_BUILDER1_H_
#define NGRAPH_TF_BRIDGE_BUILDER1_H_

#include <ostream>
#include <vector>

#include "ngraph/ngraph.hpp"

#include "ngraph_log.h"
#include "ngraph_mark_for_clustering.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"

using namespace std;
namespace ng = ngraph;
namespace tensorflow {

namespace ngraph_bridge {

// TODO: namespace detail? overload certain functions vs default args?
//TODO: make sure all comments from old builder are copied correctly.








/////////////////

class Builder1 {
  using OpMap = std::unordered_map<std::string,
                                   std::vector<std::shared_ptr<ngraph::Node>>>;

 private:
 //
  // The op map holds a mapping from TensorFlow op names (strings) to
  // vector of generated nGraph nodes.
  //
  Builder1::OpMap ng_op_map;
   


    //TODO: move GetInputNode(s) body to .cc
  Status GetInputNode(const Node* op,
                           size_t input_idx, shared_ptr<ng::Node>* result) {
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

//TODO: enable detail namespace?
//Note: namespace details prevents template recursion error during compilation
//cannot use namespace in class, so using different names
//namespace detail {
 Status detail_GetInputNodes(const Node* op,
                            size_t index) {
  return Status::OK();
}

template <typename... Arguments>
 Status detail_GetInputNodes(const Node* op,
                            size_t index, shared_ptr<ng::Node>* result,
                            Arguments&&... remaining) {
  if (result != nullptr) {
    TF_RETURN_IF_ERROR(GetInputNode(op, index, result));
  }
  return detail_GetInputNodes(op, index + 1, remaining...);
}
//}  // namespace detail

template <typename... Arguments>
 Status GetInputNodes(const Node* op, Arguments&&... remaining) {
  constexpr size_t args_len = sizeof...(Arguments);
  TF_RETURN_IF_ERROR(ValidateInputCount(op, args_len));
  //return detail::GetInputNodes(ng_op_map, op, 0, remaining...);  //TODO.. detail namespace
  return detail_GetInputNodes(op, 0, remaining...);
}


 //TODO: reconsider TranslateBinaryOp
 Status TranslateBinaryOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map,
    std::function<std::shared_ptr<ng::Node>(std::shared_ptr<ng::Node>,
                                            std::shared_ptr<ng::Node>)>
        create_binary_op) {
  std::shared_ptr<ng::Node> ng_lhs, ng_rhs;
  TF_RETURN_IF_ERROR(GetInputNodes(op, &ng_lhs, &ng_rhs));

  std::tie(ng_lhs, ng_rhs) =
      ng::builder::numpy_broadcast(std::make_pair(ng_lhs, ng_rhs));

    //TODO: enable this
  //SaveNgOp(ng_op_map, op->name(), create_binary_op(ng_lhs, ng_rhs));

  return Status::OK();
}

template <typename T>
Status TranslateBinaryOp(
    const Node* op, const std::vector<const Tensor*>& static_input_map) {
  return TranslateBinaryOp(
      op, static_input_map,
      [](std::shared_ptr<ng::Node> ng_lhs, std::shared_ptr<ng::Node> ng_rhs) {
        return make_shared<T>(ng_lhs, ng_rhs);
      });
}

 const static std::map<
    const string,
    const function<Status(const Node*, const std::vector<const Tensor*>&)>>
    TRANSLATE_OP_MAP;

  OpKernelConstruction* ctx; //TODO: do we need to save it?
  bool is_init = false; //Prevent init from running twice
  const Graph& tf_graph;
  std::vector<bool> m_input_is_static;
  vector<Node*> ordered;
  vector<const Node *> tf_params, tf_ret_vals, tf_ops;

  Status ValidateInputCount(const Node* op, size_t count);
  Status ValidateInputCountMin(const Node* op, size_t count);

  

  // helper function for populating ng_op_map
  void SaveNgOp(const std::string& op_name,
                const shared_ptr<ng::Node>& output_node);

  // TODO: write description
  Status get_input_params(const std::vector<TensorShape>&, vector<const Node*>,
                          vector<shared_ptr<ng::op::Parameter>>*);
  Status classify_nodes(const vector<Node*>&, vector<const Node*>&,
                        vector<const Node*>&, vector<const Node*>&);
  Status translate_each_op(const vector<const Node*>&);
  Status get_output_nodes(const vector<const Node*>&,
                          vector<shared_ptr<ng::Node>>&);

  template <typename T>
  void MakePadding(const std::string& tf_padding_type,
                   const ngraph::Shape& ng_image_shape,
                   const ngraph::Shape& ng_kernel_shape,
                   const ngraph::Strides& ng_strides,
                   const ngraph::Shape& ng_dilations, T& ng_padding_below,
                   T& ng_padding_above);

  template <typename T>
  void MakePadding(const std::string& tf_padding_type,
                   const ngraph::Shape& ng_image_shape,
                   const ngraph::Shape& ng_kernel_shape,
                   const ngraph::Strides& ng_strides, T& ng_padding_below,
                   T& ng_padding_above);

 public:
  // Note: we need a separate init function for this. because we cant return
  // values from constructor,
  // we cannot wrap 'classify_nodes' in 'TF_RETURN_IF_ERROR' if we did this part
  // in the constructor.
  Status init();

  // TODO: move constructor body to .cc
  Builder1(const Graph& tf_graph, OpKernelConstruction* ctx)
      : tf_graph(tf_graph), ctx(ctx) {
    //TODO: maybe we do not need to copy ctx into a pvt variable., as we do not use it later
    // TODO: move m_input_is_static construction to init??

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
        // TODO: do we need the ctx here. can we not use it?
        // macro defn:
        // https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/core/framework/op_kernel.h#L1265
        // For now, requiring builder to have access to ctx
        OP_REQUIRES_OK(ctx, GetNodeAttr(node->attrs(), "index", &index));
        if (index > max_arg_index) max_arg_index = index;
      }
    }

    m_input_is_static = std::vector<bool>(max_arg_index + 1, false);

    // Fill the vector.
    for (auto node : arg_nodes) {
      int32 index;
      OP_REQUIRES_OK(ctx, GetNodeAttr(node->attrs(), "index", &index));

      // bool is_static = false;
      for (auto edge : node->out_edges()) {
        if (edge->IsControlEdge() || !edge->dst()->IsOp()) {
          continue;
        }

        NGRAPH_VLOG(5) << "For arg " << index << " checking edge "
                       << edge->DebugString();

        if (InputIsStatic(edge->dst(), edge->dst_input())) {
          NGRAPH_VLOG(5) << "Marking edge static: " << edge->DebugString();
          // is_static = true;
          m_input_is_static[index] = true;
          break;
        }
      }

      // NGRAPH_VLOG(5) << "Marking arg " << index << " is_static: " <<
      // is_static;
      // m_input_is_static[index] = is_static;
      NGRAPH_VLOG(5) << "Marking arg " << index
                     << " is static: " << m_input_is_static[index];
    }
  }
  Status TranslateGraph(OpKernelContext* ctx,
                        std::shared_ptr<ngraph::Function>& ng_function);
};

}  // namespace ngraph_bridge

}  // namespace tensorflow

#endif
