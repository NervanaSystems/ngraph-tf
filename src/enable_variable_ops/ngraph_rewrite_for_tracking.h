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

#pragma once

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

namespace ngraph_bridge {

Status RewriteForTracking(Graph* graph, int graph_id);
Status ReplaceNGraphAssign(Graph* graph, Node* node, Node** replacement,
                           std::string node_new_name, bool just_looking,
                           bool outputs_ng_supported);
Status ReplaceNGraphVariable(Graph* graph, Node* node, Node** replacement,
                             std::string node_new_name, bool just_looking,
                             bool outputs_ng_supported);
Status ReplaceNGraphApplyGradientDescent(Graph* graph, Node* node,
                                         Node** replacement,
                                         std::string node_new_name,
                                         bool just_looking,
                                         bool outputs_ng_supported);
}  // namespace ngraph_bridge

}  // namespace tensorflow
