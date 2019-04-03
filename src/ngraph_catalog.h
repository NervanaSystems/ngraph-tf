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
        CatalogBase(){}
    friend class NGraphCatalog;
};

//using TensorID = std::tuple<string, bool, int>;

class TensorID : public std::tuple<string, bool, int> {
    string to_string();
};


 struct MyHash
{
    size_t operator()(const TensorID& t) const noexcept
    {
        string node_name; bool output; int slot;
        std::tie(node_name, output, slot) = t;
        return std::hash<string>{}(node_name + to_string(output) + to_string(slot));
        // TODO: use to_string() in case TensorID is a class instead of a type alias
    }
};


// TODO: rename it NGraphCatalogs
class NGraphCatalog {
  public:
    static CatalogBase<TensorID, string, MyHash> catalog1;
    static CatalogBase<TensorID, shared_ptr<ng::runtime::Tensor>, MyHash> catalog2;
};


}  // ngraph_bridge
}  // tensorflow

#endif