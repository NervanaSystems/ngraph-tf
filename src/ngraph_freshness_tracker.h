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
#ifndef NGRAPH_FRESHNESS_TRACKER_H_
#define NGRAPH_FRESHNESS_TRACKER_H_

#include <set>
#include "ngraph_utils.h"

#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {

namespace ngraph_bridge {
//
// A class used to track freshness of tensors for the purpose of memoization.
// Tensors are tracked according to their base pointers; thus this is only
// suitable in cases where a tensor's base pointer cannot be changed. Tensors
// internal to the nGraph bridge conform to these restrictions.
//
// General usage:
//
//   NGraphFreshnessTracker* tracker;
//   Tensor* t = ...;
//   std::shared_ptr<ngraph::Function> ng_func1 = something;
//   std::shared_ptr<ngraph::Function> ng_func2 = something_else;
//   ...
//   const void* tensor_base_ptr = (const void *)DMAHelper::base(t);
//
//   [tracker->IsFresh(tensor_base_ptr,ng_func1) will return false]
//   [tracker->IsFresh(tensor_base_ptr,ng_func2) will return false]
//
//   tracker->MarkFresh(tensor_base_ptr,ng_func1);
//                                        // TRIES to mark "t" as fresh for
//                                        // ng_func1, but has no effect
//                                        // because t has not been registered
//                                        // yet.
//
//   [tracker->IsFresh(tensor_base_ptr,ng_func1) will _still_ return false]
//   [tracker->IsFresh(tensor_base_ptr,ng_func2) will return false]
//
//   tracker->AddTensor(tensor_base_ptr); // registers "t"
//   tracker->MarkFresh(tensor_base_ptr, ng_func1); // marks "t" fresh for
//   ng_func1
//
//   [tracker->IsFresh(tensor_base_ptr,ng_func1) will return true]
//   [tracker->IsFresh(tensor_base_ptr,ng_func2) will return false]
//
//   tracker->MarkStale(tensor_base_ptr); // marks t as "stale" for all funcs
//
//   [tracker->IsFresh(tensor_base_ptr,ng_func1) will return false]
//   [tracker->IsFresh(tensor_base_ptr,ng_func2) will return false]
//
//   tracker->MarkFresh(tensor_base_ptr,ng_func1); // marks "t" fresh for
//                                                 // ng_func1
//   tracker->RemoveUser(ng_func1);                // removes all freshness
//                                                 // info for ng_func1
//
//   [tracker->IsFresh(tensor_base_ptr,ng_func1) will return false]
//   [tracker->IsFresh(tensor_base_ptr,ng_func2) will return false]
//
//   tracker->MarkFresh(tensor_base_ptr,ng_func2); // marks "t" fresh for
//                                                 // ng_func2
//   tracker->RemoveTensor(tensor_base_ptr);       // de-registers "t"
//
//   [tracker->IsFresh(tensor_base_ptr,ng_func1) will return false]
//   [tracker->IsFresh(tensor_base_ptr,ng_func2) will return false]
//
// Inside the nGraph bridge, the freshness tracker is stored as a resource in
// the ResourceMgr's default container, with the resource name
// "ngraph_freshness_tracker".
//
class NGraphFreshnessTracker : public ResourceBase {
 public:
  explicit NGraphFreshnessTracker() {}
  // Not copyable or movable.
  NGraphFreshnessTracker(const NGraphFreshnessTracker&) = delete;
  NGraphFreshnessTracker& operator=(const NGraphFreshnessTracker&) = delete;

  std::string DebugString() override { return "FreshnessTracker"; }

  // If freshness_map_ has the base_pointer, then inserts the user function into
  // its set of user functions
  void MarkFresh(const void* base_pointer,
                 std::shared_ptr<ngraph::runtime::Executable> user);

  // Checks if the freshness_map_ has the user function for base_pointer, else
  // returns false
  bool IsFresh(const void* base_pointer,
               std::shared_ptr<ngraph::runtime::Executable> user);

  // Removes all the functions for the base_pointer in the freshness_map_, i.e.
  // sets the set<ng::Function> for base_pointer to empty
  void MarkStale(const void* base_pointer);

  // Inserts the base_pointer in the map, initialises the set of functions as
  // empty set
  void AddTensor(const void* base_pointer);

  // Erases the base_pointer from the freshness_map_
  void RemoveTensor(const void* base_pointer);

  // Removes the user function from the freshness_map_
  void RemoveUser(std::shared_ptr<ngraph::runtime::Executable> user);

 private:
  // mutex protecting the freshness_map_
  mutex mu_;
  // for each base pointer (of tensor), maintains a set of ng::Functions that
  // use it
  // Each ng function in thus set is then a user of the base_pointer
  std::map<const void*, std::set<std::shared_ptr<ngraph::runtime::Executable>>>
      freshness_map_;

  ~NGraphFreshnessTracker() override {}
};
}  // namespace ngraph_bridge

}  // namespace tensorflow

#endif
