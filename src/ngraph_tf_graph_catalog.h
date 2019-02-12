/*******************************************************************************
 * Copyright 2017-2019 Intel Corporation
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
#include <string>
using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

// Tid could be tuple<int, bool, int> or string (of the form encid_isOut_slot)
template<typename Tid>
class DisjointSet {
public:
void MakeSet(Tid x);
bool Find(Tid x, Tid* found_object) // return a boolean saying if tensor is tracked or not. If found, populate found_object
{   
    //bool is_tracked = x in tracked_items
    if (is_tracked){
        // Find it
    } else {
        found_object = nullptr;
        return false;
    }
}
void Union(Tid x, Tid y);

// Once data collection is done, flatten the tree so that it is faster later
  void flatten() {
  }

~DisjointSet() {
    // Free stuff: tracked_items
}

private:
struct DisjointSetObjectWrapper {
    shared_pointer<Tid> item;
    struct DisjointSetObjectWrapper* parent
    bool isRepresentative() {return this->parent == this;}
};
std::unordered_map<Tid, DisjointSetObjectWrapper> tracked_items;
};

// TODO: sarkars: name need rethinking
// Tid is a type or class that identifies a tensor uniquely (for this purpose).
// Tid could be tuple<int, bool, int> or string (of the form encid_isOut_slot) 
template<typename Tid>
class TFGraphCatalog{
public:
  static TFGraphCatalog& getInstance(){
    static TFGraphCatalog instance;
    return instance;
  }

  int getGraphID(); // May not be needed if encapsulates get unique numbers that are different across graphs. That seems to be the case since ClusterManager is static

  bool tensorIsTracked(Tid tensor_id);

  ng::runtime::Tensor getTensorFromCatalog(Tid tensor_id){
      // if tensor is not found, throw an error
      Tid* p_ng_tensor;
      bool tensor_found = tensor_id_share.Find(tensor_id, p_ng_tensor);
  }

  void saveInCatalog(Tid tensor_id, ng::runtime::Tensor t);


private:
  int id = 0; // May not be needed if encapsulates get unique numbers that are different across graphs. That seems to be the case since ClusterManager is static

  // tuple may not be hashable. either pass a hasher, or use string
  // For this map, we could only store representative of each share group, and a one-to-one map. then we use:
  // representative = tensor_id_share.Find(tid); ngtensor = tf_tenosrid_to_ng_tensor[representative]
  // OR
  // Once we have finished collecting shared tensor info, we export that to this map, which is now many-to-one
  // In the first design, the chances of messing up the underlying shared tensor is less, bt it might take slightly longer because of the extra lookup
  // mitigation: flatten the disjointSet structure once data collection is done
  std::unordered_map<Tid, ng::runtime::Tensor> tf_tenosrid_to_ng_tensor;

  // Disjoint set that captures which tensors form a shared group
  DisjointSet<std::string> tensor_id_share;

  TFGraphCatalog()= default;
  ~TFGraphCatalog()= default;
  TFGraphCatalog(const TFGraphCatalog&)= delete;
  TFGraphCatalog& operator=(const TFGraphCatalog&)= delete;
};

}
}