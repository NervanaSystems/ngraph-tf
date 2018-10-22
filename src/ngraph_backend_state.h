/*******************************************************************************
 * Copyright 2018 Intel Corporation
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

#include <absl/base/thread_annotations.h>

#include <memory>
#include <mutex>
#include <string>
#include <tuple>

#include "ngraph/runtime/backend.hpp"

namespace tensorflow {
namespace ngraph_bridge {
namespace backend_state {

// Encapsulates the bridge's notion of the state of an execution backend.
struct State {
  std::mutex mu;  // Must be held across modifications to the state.

  // Indicates whether this is a CPU backend, which changes some
  // aspects of how tensors are passed to the backend.
  bool is_cpu GUARDED_BY(mu);

  // The backend itself.  N.B. This is managed via a shared_ptr so
  // that the caller may capture it and then drop the BackendState
  // mutex.
  std::shared_ptr<ngraph::runtime::Backend> backend GUARDED_BY(mu);
};

// The standard backend configuration environment variable name.
extern const char kStandardConfigEnvVar[];

// Indicates whether a given config string describes a CPU backend.
bool IsCPUConfig(const std::string& config) noexcept;

// Returns a convenient process-global BackendState.  Note the the
// state is minimally initialized; in particular, the backend pointer
// within the state may be empty.
State* GlobalState() noexcept;

}  // namespace backend_state
}  // namespace ngraph_bridge
}  // namespace tensorflow
