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

// The backend manager class is a singelton class that interfaces with the
// bridge to provide necessary backend

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

class TensorID;
template <class K, class V, class Hash, class KeyEqual, class Allocator>
class CatalogBase;
class NGraphCatalog;


// Consider this case: E3<--E1-->E2.
// Lets say a switch statement chooses which of E2 or E3 gets executed.
// Then we cannot have a scheme where we let each member of the group set a variable that counts how many members of the group have already used the tensor, so that the last one using it can mark the "on_device" false for the next round
// another scheme might be: the "first" op to execute in the group loads the tensor.
// But the definition if "first" might be tricky and might not be known statically
// So conclusion is that we probably cannot mark the "resetter" in a group statically

// 
template <class TID>
class Group {
 public:
  bool is_on_device() { return on_device; }
  bool tid_in_group(TID t) {
    return group_members.find(t) != group_members.end();
  }

 private:
  bool on_device = false;
  std::set<TID> group_members;
  
};

// TensorID classes are expected to provide a to_string()
// and a hash functor (if necessary)
class TensorID : public std::tuple<string, bool, int> {
 public:
  // Note, we are inheriting constructors here. which means this class should
  // not have any data member
  // If new data members are needed, then new constructors need to be written
  // The intention here is to souped-up the tuple class with some user friendly
  // functions
  using tuple::tuple;

  string to_string() const {
    string node_name;
    bool output;
    int slot;
    std::tie(node_name, output, slot) = *this;
    return node_name + "_" + std::to_string(output) + "_" +
           std::to_string(slot);
  }

  string get_node_name() { return std::get<0>(*this); }

  bool get_output() { return std::get<1>(*this); }

  int get_slot() { return std::get<2>(*this); }

  struct TensorIDHash {
    size_t operator()(const TensorID& t) const noexcept {
      return std::hash<string>{}(t.to_string());
    }
  };
};

template <class K, class V, class Hash = std::hash<K>,
          class KeyEqual = std::equal_to<K>,
          class Allocator = std::allocator<std::pair<const K, V>>>
class CatalogBase : public std::unordered_map<K, V, Hash, KeyEqual, Allocator> {
 public:
  // This is not needed perhaps. since its a dictionary [] is enough
  void AddToCatalog(V val, K key);
  bool ExistsInCatalog(K key);
  // Making CatalogBase unconstructable, except by NGraphCatalog
 private:
  CatalogBase() {}
  friend class NGraphCatalog;
};

// TODO: rename it NGraphCatalogs
class NGraphCatalog {
 public:
  // TODO: catalog1 and catalog2 are shrestha's catalogs
  static CatalogBase<TensorID, string, TensorID::TensorIDHash> catalog1;
  static CatalogBase<TensorID, shared_ptr<ng::runtime::Tensor>,
                     TensorID::TensorIDHash>
      catalog2;

  static CatalogBase<TensorID, string, TensorID::TensorIDHash>
      non_modifiable_tensor_catalog;
};

}  // ngraph_bridge
}  // tensorflow

#endif