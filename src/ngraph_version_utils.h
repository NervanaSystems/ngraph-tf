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
#ifndef NGRAPH_TF_BRIDGE_VERSION_UTILS_H_
#define NGRAPH_TF_BRIDGE_VERSION_UTILS_H_

#include "tensorflow/core/public/version.h"

#define TF_VERSION_GEQ(REQ_TF_MAJ_VER, REQ_TF_MIN_VER) \
  ((TF_MAJOR_VERSION > REQ_TF_MAJ_VER) ||              \
   ((TF_MAJOR_VERSION == REQ_TF_MAJ_VER) &&            \
    (TF_MINOR_VERSION >= REQ_TF_MIN_VER)))

#define TF_VERSION_GEQ_1_11 TF_VERSION_GEQ(1, 11)
#endif  // NGRAPH_TF_BRIDGE_VERSION_UTILS_H_