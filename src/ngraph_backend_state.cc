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

#include "ngraph_backend_state.h"

namespace tensorflow {
namespace ngraph_bridge {
namespace backend_state {

const char kStandardConfigEnvVar[] = "NGRAPH_TF_BACKEND";

bool IsCPUConfig(const std::string& config) noexcept {
  // nGraph backend configuration strings start with the library name,
  // optionally followed by a ':' and a comma-separated sequence of
  // attributes.  So to discover whether we're using the CPU backend,
  // we match the config up to the first ':' (if any).
  auto colon = config.find(':');
  if (colon == std::string::npos) {
    return config.compare("CPU") == 0;
  }
  return config.compare(0, colon, "CPU") == 0;
}

State* GlobalState() noexcept {
  static State state;
  return &state;
}

}  // namespace backend_state
}  // namespace ngraph_bridge
}  // namespace tensorflow
