/*******************************************************************************
 * Copyright 2019-2020 Intel Corporation
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
#ifndef NGRAPH_TF_ENTER_IN_CATALOG_H_
#define NGRAPH_TF_ENTER_IN_CATALOG_H_
#pragma once

#include "tensorflow/core/graph/graph.h"

#include "ngraph/ngraph.hpp"
#include "ngraph_catalog.h"


using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

// 1. Populate the input_variable_map
// 2. Attach Graph Ids to the node
Status EnterInCatalog(Graph* graph, int graph_id);

}  // ngraph_bridge
}  // tensorflow

#endif
