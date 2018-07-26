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

#include <utility>

#include "ngraph_builder.h"
#include "ngraph_log.h"

namespace ngraph_bridge {

template <typename T>
std::pair<const std::vector<T>&, std::vector<size_t>&> src_dst(
    const std::vector<T>& src, std::vector<size_t>& dst) {
  return std::pair<const std::vector<T>&, std::vector<size_t>&>(src, dst);
}

template <size_t a, size_t b, size_t c, size_t d>
void Reshape(std::shared_ptr<ng::Node>& ng_node) {
  static_assert(a < 4 && b < 4 && c < 4 && d < 4,
                "Number of dimensions cannot exceed 4");
  static_assert(a != b && a != c && a != d && b != c && b != d && c != d,
                "Dimensions indices cannot be equal");
  auto& s = ng_node->get_shape();
  ng::Shape reshaped_shape{s[a], s[b], s[c], s[d]};
  NGRAPH_VLOG(3) << "reshaped_shape: " << ng::join(reshaped_shape);
  ng_node = std::make_shared<ng::op::Reshape>(
      ng_node, ng::AxisVector{a, b, c, d}, reshaped_shape);
}

void NhwcToNgraph() {}

template <typename... Arguments, typename T>
void NhwcToNgraph(
    const std::pair<const std::vector<T>&, std::vector<size_t>&>& param,
    Arguments&&... remaining) {
  param.second[0] = param.first[1];
  param.second[1] = param.first[2];
  NhwcToNgraph(remaining...);
}
template <typename... Arguments>
void NhwcToNgraph(std::shared_ptr<ng::Node>& ng_node,
                  Arguments&&... remaining) {
  Reshape<0, 3, 1, 2>(ng_node);
  NhwcToNgraph(remaining...);
}

void NchwToNgraph() {}

template <typename... Arguments, typename T>
void NchwToNgraph(
    const std::pair<const std::vector<T>&, std::vector<size_t>&>& param,
    Arguments&&... remaining) {
  param.second[0] = param.first[2];
  param.second[1] = param.first[3];
  NchwToNgraph(remaining...);
}

template <typename... Arguments>
void TensorflowToNgraph(bool is_nhwc, std::shared_ptr<ng::Node>& ng_input,
                        Arguments&&... remaining) {
  if (is_nhwc) {
    NhwcToNgraph(ng_input, remaining...);
  } else {
    NchwToNgraph(remaining...);
  }
}

void NgraphToTensorflow(bool is_nhwc, std::shared_ptr<ng::Node>& ng_node) {
  if (!is_nhwc) {
    return;
  }
  Reshape<0, 2, 3, 1>(ng_node);
}
}
