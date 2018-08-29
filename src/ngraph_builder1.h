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

//#include "ngraph_conversions.h"

#include "ngraph/ngraph.hpp"

#include "ngraph_log.h"
#include "ngraph_mark_for_clustering.h"

// TODO: remove headers if not needed
//#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/algorithm.h"
//#include "tensorflow/core/graph/graph.h"

using namespace std;
namespace ng = ngraph;
namespace tensorflow {

namespace ngraph_bridge {

// TODO: make sure all comments from old builder are copied correctly.
// TODO: use camelcase, snakecase appropriately
// TODO add TF_RETURN_IF_ERROR where necessary

/////////////////

class Builder1 {
  using VectNg = std::vector<shared_ptr<ng::Node>>;
  using TranslatorFn = function<Status(
      const Node*, VectNg&, const std::vector<const Tensor*>&, VectNg&)>;

 public:
  // Note: in case we want to get rid of Initialize,
  // pass a OpKernelConstruction* ctx to the constructor,
  // and use OP_REQUIRES_OK instead of TF_RETURN_IF_ERROR
  // This gets rid of Initialize() as it can handle errors in construction
  // But in the case of overloaded constructor that does not accept a ctx,
  // which is used for OpExecuter test class, we cannot handle error during
  // construction.
  // Hence keeping the Initialize() function
  Builder1(const Graph& tf_graph,
           OpKernelConstruction* ctx)  // TODO make ctx const?
      : tf_graph(tf_graph) {}

  Builder1(const Graph& tf_graph) : Builder1(tf_graph, nullptr) {}

  Status TranslateGraph(const std::vector<TensorShape>&,
                        const std::vector<const Tensor*>&,
                        shared_ptr<ng::Function>&);

  Status TranslateGraph(OpKernelContext* ctx,
                        std::shared_ptr<ngraph::Function>& ng_function);

 private:
  //
  // The op map holds a mapping from TensorFlow op names (strings) to
  // vector of generated nGraph nodes.
  //
  std::unordered_map<std::string, VectNg> ng_op_map;

  const static std::map<const string, vector<int>> INPUT_INDEX_MAP;

  bool is_initialized = false;  // Prevent Initialize from running twice
  const Graph& tf_graph;
  std::vector<bool> m_input_is_static;
  vector<Node*> ordered;
  vector<const Node *> tf_params, tf_ret_vals, tf_ops;

  // A map from Tf op type_string to a TranslateOp
  const static std::map<const string, Builder1::TranslatorFn> TRANSLATE_OP_MAP;
  // Given a TF node, return its corresponding TranslateOp function and required input indexes
  // A wrapper for TRANSLATE_OP_MAP
  Status GetOpTranslationRequirements(const Node*, Builder1::TranslatorFn&,
                                      vector<int>&);

  Status GetInputNode(const Node*, size_t, shared_ptr<ng::Node>*);

  // Classify a list of TF nodes into _Arg (input), _Retval (output) and other
  // nodes
  Status ClassifyNodes(const vector<Node*>&, vector<const Node*>&,
                       vector<const Node*>&, vector<const Node*>&);

  // Given the input shapes and a list of TF _Arg nodes, create corresponding
  // nGraph parameters. Also populate the ng_op_map
  Status GetInputParams(const std::vector<TensorShape>&, vector<const Node*>,
                        vector<shared_ptr<ng::op::Parameter>>&);
  // Given a TF node, retrieve its corresponding nGraph nodes (using ng_op_map),
  // then call the appropriate TranslateOp function
  Status TranslateEachOp(const vector<const Node*>&,
                         const std::vector<const Tensor*>&);
  // After each TF op has been translated, find the nGraph nodes corresponding
  // to the _Retval nodes
  Status GetOutputNodes(const vector<const Node*>&, VectNg&);

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
};

}  // namespace ngraph_bridge

}  // namespace tensorflow

#endif
