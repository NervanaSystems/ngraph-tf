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
#include "tensorflow/core/lib/core/errors.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph_catalog.h"
#include "ngraph_log.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

unordered_map<string, string> NGraphCatalog::input_variable_map_;

string NGraphCatalog::CreateNodeKey(int graph_id, string node_name,
                                    int inp_index) {
  return to_string(graph_id) + "_" + node_name + "_" + to_string(inp_index);
}

string NGraphCatalog::GetInputSharedName(int graphid, string node_name,
                                         int input_index) {
  std::string node_key =
      NGraphCatalog::CreateNodeKey(graphid, node_name, input_index);
  return NGraphCatalog::input_variable_map_[node_key];
}

void NGraphCatalog::AddCatalog(string key, string val) {
  NGraphCatalog::input_variable_map_[key] = val;
}

bool NGraphCatalog::ExistsInCatalog(string key) {
  auto itr = NGraphCatalog::input_variable_map_.find(key);
  return itr != NGraphCatalog::input_variable_map_.end();
}

bool NGraphCatalog::ExistsInCatalog(int graphid, string node_name,
                                    int input_index) {
  return NGraphCatalog::ExistsInCatalog(
      NGraphCatalog::CreateNodeKey(graphid, node_name, input_index));
}

}  // ngraph_bridge
}  // tensorflow
