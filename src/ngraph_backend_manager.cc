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

#include "ngraph_backend_manager.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

// initialize backend manager
string BackendManager::ng_backend_name_ = "CPU";
mutex BackendManager::ng_backend_name_mutex_;
map<string, Backend*> BackendManager::ng_backend_map_;
mutex BackendManager::ng_backend_map_mutex_;

// void BackendManager::lock(){}
// void BackendManager::unlock(){}

Status BackendManager::SetBackendName(const string& backend_name) {}

void BackendManager::CreateBackendIfDoesNotExist(const string& backend_name) {
  std::lock_guard<std::mutex> lock(BackendManager::ng_backend_map_mutex_);
  auto itr = BackendManager::ng_backend_map_.find(backend_name);
  // if backend does not exist create it
  if (itr == BackendManager::ng_backend_map_.end()) {
    Backend* bend = new Backend;
    std::unique_ptr<ng::runtime::Backend> bend_ptr =
        ng::runtime::Backend::create(backend_name);
    bend->backend_ptr = std::move(bend_ptr);
    BackendManager::ng_backend_map_[backend_name] = bend;
  }
}

// Returns a backend pointer of the type specified by the backend name
ng::runtime::Backend* BackendManager::GetBackend(const string& backend_name) {
  return BackendManager::ng_backend_map_.at(backend_name)->backend_ptr.get();
}

// LockBackend
/*static*/ void BackendManager::LockBackend(const string& backend_name) {
  BackendManager::ng_backend_map_.at(backend_name)->backend_mutex.lock();
}

// UnlockBackend
/*static*/ void BackendManager::UnlockBackend(const string& backend_name) {
  BackendManager::ng_backend_map_.at(backend_name)->backend_mutex.unlock();
}

// Returns the nGraph supported backend names
Status BackendManager::GetSupportedBackendNames(vector<string>& backend_names) {
}

}  // namespace ngraph_bridge
}  // namespace tensorflow