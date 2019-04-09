/*******************************************************************************
 * Copyright 2019 Intel Corporation
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
#ifndef NGRAPH_TF_REPLACE_OP_UTILITIES_H_
#define NGRAPH_TF_REPLACE_OP_UTILITIES_H_

#pragma once

#include "tensorflow/core/graph/graph.h"

using namespace std;
namespace tensorflow {

namespace ngraph_bridge {

Status ReplaceApplyGradientDescent(Graph* graph, Node* node, Node** replacement,
                                   std::string replacement_node_name,
                                   std::string replacement_op_type,
                                   bool just_looking, bool outputs_ng_supported,
                                   int graph_id);

Status ReplaceAssign(Graph* graph, Node* node, Node** replacement,
                     std::string replacement_node_name,
                     std::string replacement_op_type, bool just_looking,
                     bool outputs_ng_supported, int graph_id);

Status ReplaceVariable(Graph* graph, Node* node, Node** replacement,
                       std::string replacement_node_name,
                       std::string replacement_op_type, bool just_looking,
                       bool outputs_ng_supported, int graph_id);

}  // namespace ngraph_bridge

}  // namespace tensorflow

#endif  // NGRAPH_TF_REPLACE_OP_UTILITIES_H_