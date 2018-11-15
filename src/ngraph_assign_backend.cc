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

#pragma once

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

namespace ngraph_bridge {

// If there are only certain type of ops that should be assigned a backend we
// can add those checks here For e.g. : only ops placed on 'device' CPU etc.
Status CanAssignBackend(Node* node, bool& can_assign_backend) {
  can_assign_backend = true;
  return Status::OK();
}

// Assigns the currently set backend to all the ops
Status AssignBackend(Graph* graph) {}

Status GetNodeBackend(const Node* node, string* backend_name) {
  // TODO(amprocte): move attr name to a constant
  Status s = GetNodeAttr(node->attrs(), "_ngraph_backend", backend_name);
  if (s != Status::OK()) {
    *backend_name = "NotSet";
  }
  return s;
}

}  // namespace ngraph_bridge
}  // namespace tensorflow