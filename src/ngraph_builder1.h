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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/algorithm.h"

using namespace std;
namespace ng = ngraph;
namespace tensorflow {

namespace ngraph_bridge {

// TODO (sarkars): make sure all comments from old builder are copied correctly.

// The purpose of Builder is to accept a TF graph and convert it to a
// equivalent ngraph fucntion. To that end, it acts as an interface to
// a library of 'TranslateOps' functions. These functions are defined
// in ngraph_translateops.h. Builder just goes through each TF op and
// calls the appropriate 'TranslatorFn'. It does not implement the per-op
// translation, the heavy-lifting being done by ngraph_translateops.h;
// It merely does some bookkeeping, such as process the nodes in correct
// order, keep track of and retrieve parent nGraph nodes etc.

class Builder1 {
  using VectNg = std::vector<shared_ptr<ng::Node>>;
  using TranslatorFn = function<Status(
      const Node*, VectNg&, const std::vector<const Tensor*>&, VectNg&)>;

 public:
  // Note: in case we want to get rid of Initialize, pass an
  // OpKernelConstruction* ctx to the constructor, and use OP_REQUIRES_OK
  // instead of TF_RETURN_IF_ERROR. This gets rid of Initialize() as it
  // can handle errors in construction.
  // But in the case of overloaded constructor that does not accept a ctx,
  // which is used for OpExecuter test class, we cannot handle error during
  // construction.
  // Hence keeping the Initialize() function

  // This constructor is for actual use (encapsulate_op)
  Builder1(const Graph& tf_graph, OpKernelConstruction* ctx)
      : tf_graph(tf_graph) {}

  // And this version is for tests (OpExecuter)
  Builder1(const Graph& tf_graph) : Builder1(tf_graph, nullptr) {}

  // TranslateGraph is overloaded. This is for actual use (encapsulate_op)
  Status TranslateGraph(OpKernelContext* ctx,
                        std::shared_ptr<ngraph::Function>& ng_function);

  // And this version is for tests (OpExecuter)
  Status TranslateGraph(const std::vector<TensorShape>&,
                        const std::vector<const Tensor*>&,
                        shared_ptr<ng::Function>&);

 private:
  // The op map holds a mapping from TensorFlow op names (strings) to
  // vector of generated nGraph nodes. Since we process nodes in toposort
  // order, it is guaranteed when processing a TF node, its parents
  // will already have been processed and safely stowed in ng_op_map
  std::unordered_map<std::string, VectNg> ng_op_map;

  // Prevent Initialize from running twice
  bool is_initialized = false;
  const Graph& tf_graph;

  // An array that will be useful in creating the static_input_map
  std::vector<bool> m_input_is_static;

  // Vectors containing TF nodes in 3 bins: inputs, outputs, actual ops
  vector<const Node *> tf_params, tf_ret_vals, tf_ops;

  // This map tells us which inputs to read for a particular node. If no
  // information is present explicitly in the map, we read all inputs
  // from 0 to num_inputs-1
  const static std::map<const string, vector<int>> INPUT_INDEX_MAP;

  // A map from Tf op type_string to a TranslateOp
  const static std::map<const string, Builder1::TranslatorFn> TRANSLATE_OP_MAP;

  // Given a TF node, return its corresponding TranslateOp function and required
  // input indexes. A wrapper for TRANSLATE_OP_MAP and INPUT_INDEX_MAP
  Status GetOpTranslationRequirements(const Node*, Builder1::TranslatorFn&,
                                      vector<int>&);

  // Given a TF node, an index i, it returns the ith nGraph input
  Status GetInputNode(const Node*, size_t, shared_ptr<ng::Node>*);

  // Classify a list of TF nodes into _Arg (input), _Retval (output) and other
  // nodes. Used to populate tf_params, tf_ret_vals, tf_ops
  Status ClassifyNodes(const vector<Node*>&);

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

  // Since the constructor does not return Status, delegating its job to a
  // separate function that is evaluated lazily and only once.
  Status Initialize();
};

}  // namespace ngraph_bridge

}  // namespace tensorflow

#endif
