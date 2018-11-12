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

// The backend manager class is a singelton class that interfaces with the
// bridge to provide necessary backend

#ifndef NGRAPH_TF_BRIDGE_BACKEND_MANAGER_H_
#define NGRAPH_TF_BRIDGE_BACKEND_MANAGER_H_

#include <ostream>
#include <vector>

#include "ngraph/ngraph.hpp"
#include <absl/base/thread_annotations.h>
#include "ngraph_log.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

struct Backend {
  unique_ptr<ng::runtime::Backend> backend_ptr;
  mutex backend_mutex;
};


class BackendManager {
 public:
  // Returns the backend name currently set
  static Status GetCurrentlySetBackendName(string& backend_name) {
    backend_name = ng_backend_name_;
    return Status::OK();
  };

  static void CreateBackendIfDoesNotExist(const string& backend_name);

  static Status SetBackendName(const string& backend_name);
  // Returns a backend pointer of the type specified by the backend name
  static ng::runtime::Backend* GetBackend(const string& backend_name);

  // LockBackend
  static void LockBackend(const string& backend_name);

  // UnlockBackend
  static void UnlockBackend(const string& backend_name);

  // Returns the nGraph supported backend names
  static Status GetSupportedBackendNames(vector<string>& backend_names);

 //private:
  static string ng_backend_name_;
  static mutex ng_backend_name_mutex_;
  static map<string, Backend*> ng_backend_map_;
  static mutex ng_backend_map_mutex_;
};

}  // namespace ngraph_bridge
}  // namespace tensorflow

#endif
// NGRAPH_TF_BRIDGE_BACKEND_MANAGER_H