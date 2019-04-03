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

template <class K, class V, class Hash = std::hash<K>>
class CatalogBase {
 public:
  void AddToCatalog(V val, K key);
  bool ExistsInCatalog(K key);

 private:
  unordered_map<K, V, Hash> internal_storage;
  // Making CatalogBase unconstructable, except by NGraphCatalog
  CatalogBase() {}
  friend class NGraphCatalog;
};

class TensorID : public std::tuple<string, bool, int> {
 public:
  string to_string() const {
    string node_name;
    bool output;
    int slot;
    std::tie(node_name, output, slot) = *this;
    return node_name + "_" + std::to_string(output) + "_" +
           std::to_string(slot);
  }

  struct TensorIDHash {
    size_t operator()(const TensorID& t) const noexcept {
      return std::hash<string>{}(t.to_string());
    }
  };
};

// TODO: rename it NGraphCatalogs
class NGraphCatalog {
 public:
  static CatalogBase<TensorID, string, TensorID::TensorIDHash> catalog1;
  static CatalogBase<TensorID, shared_ptr<ng::runtime::Tensor>,
                     TensorID::TensorIDHash>
      catalog2;
};

template <class K, class V, class Hash = std::hash<K>,
          class KeyEqual = std::equal_to<K>,
          class Allocator = std::allocator<std::pair<const K, V>>>
class CatalogBase2
    : private std::unordered_map<K, V, Hash, KeyEqual, Allocator> {
 public:
  // This is not needed perhaps. since its a dictionary [] is enough
  void AddToCatalog(V val, K key);
  bool ExistsInCatalog(K key);
  friend class NGraphCatalog2;
};

class NGraphCatalog2 {
 public:
  // TODO: catalog1 and catalog2 are shrestha's catalogs
  static CatalogBase2<TensorID, string, TensorID::TensorIDHash> catalog1;
  static CatalogBase2<TensorID, shared_ptr<ng::runtime::Tensor>,
                      TensorID::TensorIDHash>
      catalog2;

  static CatalogBase2<TensorID, string, TensorID::TensorIDHash>
      non_modifiable_tensor_catalog;
};

}  // ngraph_bridge
}  // tensorflow

#endif