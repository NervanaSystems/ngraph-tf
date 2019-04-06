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

#ifndef NGRAPH_TF_CATALOG_H_
#define NGRAPH_TF_CATALOG_H_

#include <atomic>
#include <mutex>
#include <ostream>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph_log.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

class NGraphCatalog {
 public:
  // Map keeps track of nodes whose input is a variable tensor
  // Will be used by Assign/Optimizers and NGraphEncapsulate Op
  // Map of
  // Key string : GraphId + _ + nodename + : + input_index
  // Value : variable shared_name
  // LOCK?
  static unordered_map<string, string> input_variable_map_;

  // Map keeps track of nodes whose input is a tensor computed by NGraph
  // For e.g. if the value to be assigned was computed by NGraphEncapsulate Op
  // Will be used by Assign/Optimizers
  // Map of
  // Key
  //   when op index ==0
  //      string : GraphId + _ + nodename
  //   otherwise
  //     string : GraphId + _ + nodename + : + output_index
  // Value : shared_ptr<ng::runtime::Tensor>
  static map<string, shared_ptr<ng::runtime::Tensor>> output_tensor_map_;

  // Map keeps track of output indexes of NGraphEncapsulate Op
  // that will be used by TF Nodes or other NGraphEncapsulate Op
  // Will be used by NGraphEncapsulateOP
  // Map of
  // Key
  //  string : nodename (nGraphEncapsulateOp name)
  // Value : Set of indices
  static unordered_map<string, unordered_set<int>> ng_encap_output_copy_map_;

  // Utility Functions for the data structures
  // Functions for Encapsulate Output Copy
  static void AddToEncapOutputCopyCatalog(string key, unordered_set<int> val);
  static bool EncapOutputNeedsCopy(string key, int index);
  static unordered_set<int> GetEncapOutputIndexesNeedsCopy(string key);

  // Functions relating Variable Output Copy
  static string GetInputSharedName(int graphid, string node_name,
                                   int input_index);
  static string CreateNodeKey(int graph_id, string node_name, int inp_index);

  static void AddCatalog(string key, string val);

  static bool ExistsInCatalog(string key);
  static bool ExistsInCatalog(int graphid, string node_name, int input_index);

  static void AddOutputCatalog(string key,
                               shared_ptr<ng::runtime::Tensor> ng_val);
  static bool ExistsInOutputCatalog(string key);
  static bool ExistsInOutputCatalog(int graphid, string node_name,
                                    int input_index);

  static shared_ptr<ng::runtime::Tensor> GetNgTensorFromOutputCatalog(
      string key);
  static void DeleteTensorFromEncapOutputCatalog(string key);
};

}  // ngraph_bridge
}  // tensorflow

#endif
