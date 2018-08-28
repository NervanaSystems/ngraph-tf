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

// TODO: overload certain functions (makepadding) vs default args?
// TODO: make sure all comments from old builder are copied correctly.
// TODO: use camelcase, snakecase appropriately
// TODO add TF_RETURN_IF_ERROR where necessary

/////////////////

class Builder1 {
  //TODO: Do we need to typedef OpMap, as it is used only once
  using OpMap = std::unordered_map<std::string,
                                   std::vector<std::shared_ptr<ngraph::Node>>>;
  using TranslatorFn = function<Status(
      const Node*, const std::vector<shared_ptr<ng::Node>>&,
      const std::vector<const Tensor*>&, vector<shared_ptr<ng::Node>>&)>;
  //using DispatchTable =
  //    const std::map<const string,
  //                   std::pair<Builder1::TranslatorFn, vector<int>>>;

 private:

  //
  // The op map holds a mapping from TensorFlow op names (strings) to
  // vector of generated nGraph nodes.
  //
  Builder1::OpMap ng_op_map;

  
  

  const static std::map<const string, Builder1::TranslatorFn> TRANSLATE_OP_MAP;

  const static std::map<const string, vector<int>> INPUT_INDEX_MAP;

  OpKernelConstruction* ctx;  // TODO: is this required? required in
                              // OP_REQUIRES_OK is used instead of
                              // TF_RETURN_IF_ERROR in Initialize
  bool is_initialized = false;  // Prevent init from running twice
  const Graph& tf_graph;
  std::vector<bool> m_input_is_static;
  vector<Node*> ordered;
  vector<const Node *> tf_params, tf_ret_vals, tf_ops;

  Status GetOpTranslationRequirements(const Node*, Builder1::TranslatorFn&, vector<int>&);

  Status GetInputNode(const Node*, size_t, shared_ptr<ng::Node>*);

  // helper function for populating ng_op_map
  // TODO: no one to use it (except 1 place). delete?
  void SaveNgOp(const std::string& op_name,
                const shared_ptr<ng::Node>& output_node);

  // TODO: write description
  Status GetInputParams(const std::vector<TensorShape>&, vector<const Node*>,
                        vector<shared_ptr<ng::op::Parameter>>*);
  Status ClassifyNodes(const vector<Node*>&, vector<const Node*>&,
                       vector<const Node*>&, vector<const Node*>&);
  Status TranslateEachOp(const vector<const Node*>&,
                         const std::vector<const Tensor*>&);
  Status GetOutputNodes(const vector<const Node*>&,
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

  Status Initialize();

 public:

  Builder1(const Graph& tf_graph,
           OpKernelConstruction* ctx)  // TODO make ctx const?
      : tf_graph(tf_graph),
        ctx(ctx) {}

  Builder1(const Graph& tf_graph) : tf_graph(tf_graph) {}

  Status TranslateGraph(
    const std::vector<TensorShape>&,
    const std::vector<const Tensor*>&,
    shared_ptr<ng::Function>&);

  Status TranslateGraph(OpKernelContext* ctx,
                        std::shared_ptr<ngraph::Function>& ng_function);
};

}  // namespace ngraph_bridge

}  // namespace tensorflow

#endif
