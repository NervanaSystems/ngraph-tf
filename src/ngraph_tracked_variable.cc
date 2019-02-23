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

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/strings/strcat.h"

#include "tensorflow/core/platform/default/logging.h"

#include "ngraph/runtime/backend.hpp"
#include "ngraph_backend_manager.h"
#include "ngraph_freshness_tracker.h"
#include "ngraph_utils.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

//
// Forked from tensorflow:tensorflow/core/kernels/variable_ops.{cc,h}
// and tensorflow:tensorflow/core/ops/state_ops.cc.
//

// Resource stored by variables in the resource manager
// (legacy, ref-style version).
//
// (Changes: Renamed from LegacyVar, modified to take a TensorShape in
// constructor.)

// THIS CLASS IS NOT BEING USED ANYWHERE
class NGraphVar : public ResourceBase {
 public:
  explicit NGraphVar(DataType dtype, TensorShape shape, string BackendName)
      : tf_tensor_(dtype, shape), ng_backend_name_(BackendName) {
    // TF datatype to nGraph element type
    ng::element::Type ng_element_type;
    TFDataTypeToNGraphElementType(dtype, &ng_element_type);

    // TF TensorShape to nGraphShape
    ng::Shape ng_shape(shape.dims());
    for (int j = 0; j < shape.dims(); ++j) {
      ng_shape[j] = shape.dim_size(j);
    }

    NGRAPH_VLOG(1) << "Created ng shape and ng element";

    // Create Backend
    BackendManager::CreateBackend(ng_backend_name_);
    ng::runtime::Backend* op_backend =
        BackendManager::GetBackend(ng_backend_name_);
    NGRAPH_VLOG(1) << "Created ng backend";

    // Create nGTensor
    //void* current_src_ptr = (void*)DMAHelper::base(&tf_tensor_);
    ng_tensor_ =
        op_backend->create_tensor(ng_element_type, ng_shape);
    NGRAPH_VLOG(1) << "Created ng tensor";
  }
  // Not copyable or movable.
  NGraphVar(const NGraphVar&) = delete;
  NGraphVar& operator=(const NGraphVar&) = delete;

  mutex* mu() { return &mu_; }
  Tensor* tensor() { return &tf_tensor_; }
  shared_ptr<ngraph::runtime::Tensor> ng_tensor() { return ng_tensor_; };

  string DebugString() override {
    return strings::StrCat(DataTypeString(tf_tensor_.dtype()), "/",
                           tf_tensor_.shape().DebugString());
  }

 private:
  mutex mu_;
  Tensor tf_tensor_;
  shared_ptr<ngraph::runtime::Tensor> ng_tensor_;
  string ng_backend_name_;

  ~NGraphVar() override {}
};

/* -------------------------------------------------
//
// NGraphVariableOp
//
---------------------------------------------------*/
class NGraphVariableOp : public OpKernel {
 public:
  explicit NGraphVariableOp(OpKernelConstruction* context);
  ~NGraphVariableOp() override;
  void Compute(OpKernelContext* ctx) override;

 private:
  int graph_id;
  DataType dtype_;
  TensorShape shape_;
  bool just_looking_;
  bool convert_to_tf_tensor; 
  NGraphFreshnessTracker* tracker_;
  string ng_backend_name_;
  mutex init_mu_;
  ContainerInfo cinfo_ GUARDED_BY(init_mu_);
  bool initialized_ GUARDED_BY(init_mu_){false};

  TF_DISALLOW_COPY_AND_ASSIGN(NGraphVariableOp);
};

// RemoveRefType
// inline DataType RemoveRefType(DataType dtype) {
//   DCHECK(IsRefType(dtype));
//   return static_cast<DataType>(dtype - kDataTypeRefOffset);
// }

NGraphVariableOp::NGraphVariableOp(OpKernelConstruction* context)
    : OpKernel(context),
      tracker_(nullptr),
      just_looking_(false),
      ng_backend_name_("CPU"),  // can be through an attribute like encapsulate
      dtype_(RemoveRefType(context->output_type(0))) {
  OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
  OP_REQUIRES_OK(context, context->GetAttr("just_looking", &just_looking_));
  OP_REQUIRES_OK(context, context->GetAttr("_convert_to_tf_tensor", &convert_to_tf_tensor));
  cout << "IN tracked variable getting convert_to_tf attribtue " << convert_to_tf_tensor <<endl;

  NGRAPH_VLOG(5) << def().name() << ": just looking? " << just_looking_;
  NGRAPH_VLOG(1) << "Constructor " << def().name() << ": just looking? "
                 << just_looking_;
}

NGraphVariableOp::~NGraphVariableOp() { tracker_->Unref(); }

// (Changes: Renamed from VariableOp, modified to pass TensorShape to NGraphVar
// constructor.)
void NGraphVariableOp::Compute(OpKernelContext* ctx) {
  NGRAPH_VLOG(1) << "Compute " << def().name();
  mutex_lock l(init_mu_);
  if (!initialized_) {
    OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(),
                                    true /* use name() */));
    initialized_ = true;
  }
  auto creator = [this](NGraphVar** var) {
    *var = new NGraphVar(dtype_, shape_, ng_backend_name_);
    //(*var)->tensor()->set_shape(shape_);
    BackendManager::ng_variable_map_[def().name()] = (*var)->ng_tensor();
    NGRAPH_VLOG(1)<<"In Variable Compute "<<def().name();
    NGRAPH_VLOG(1)<<"Is Null "<< (BackendManager::ng_variable_map_[def().name()]==NULL? "Yes" : "No");
    return Status::OK();
  };

  // If "container" has a resource "name", returns it in
  // "*resource". Otherwise, invokes creator() to create the resource.
  // The caller takes the ownership of one ref on "*resource".
  //

  // Here uses the Resource Manager's default container
  NGraphVar* var;
  OP_REQUIRES_OK(ctx, cinfo_.resource_manager()->LookupOrCreate<NGraphVar>(
                          cinfo_.container(), cinfo_.name(), &var, creator));

  NGRAPH_VLOG(1)<<"Print ng-tensor";
  PrintNGTensor(var->ng_tensor());

  auto tf_tensor = var->tensor();
   
   auto ng_tensor_to_assign = var->ng_tensor();
   if(convert_to_tf_tensor){
     cout << "copy to tf tensor " << endl;
     void* tf_src_ptr = (void*)DMAHelper::base(tf_tensor);
    ng_tensor_to_assign->read(tf_src_ptr, 0, ng_tensor_to_assign->get_element_count() * ng_tensor_to_assign->get_element_type().size());
  }

  NGRAPH_VLOG(1) << "Print tf-tensor";
  PrintTFTensor(*(var->tensor()));

  // Output a reference to our tensor, so it may be updated.
  //
  // As long as the resource manager hasn't been cleared the ref we return
  // here is valid because it owns a ref on var.

  // Mark the underlying tensor as stale. TODO(amprocte): Make this
  // conditional on whether any reader is taking in a reference. More
  // conservative condition that would work for now: invalidate if any
  // reader is not NGraphEncapsulateOp.
  auto t_creator = [this](NGraphFreshnessTracker** tracker) {
    *tracker = new NGraphFreshnessTracker();
    return Status::OK();
  };
  if (tracker_ == nullptr) {
    if (NGRAPH_VLOG_IS_ON(5)) {
      NGRAPH_VLOG(5) << "Variable " << ctx->op_kernel().name()
                     << ": getting tracker";
    }
    OP_REQUIRES_OK(
        ctx, ctx->resource_manager()->LookupOrCreate<NGraphFreshnessTracker>(
                 ctx->resource_manager()->default_container(),
                 "ngraph_freshness_tracker", &tracker_, t_creator));
    if (NGRAPH_VLOG_IS_ON(5)) {
      NGRAPH_VLOG(5) << "Variable " << ctx->op_kernel().name()
                     << ": got tracker";
    }
  }

  if (NGRAPH_VLOG_IS_ON(5)) {
    NGRAPH_VLOG(5) << "Variable " << ctx->op_kernel().name() << ": adding "
                   << DMAHelper::base(var->tensor());
  }
  tracker_->AddTensor(DMAHelper::base(var->tensor()));
  if (NGRAPH_VLOG_IS_ON(5)) {
    NGRAPH_VLOG(5) << "Variable " << ctx->op_kernel().name() << ": added "
                   << DMAHelper::base(var->tensor());
  }

  if (!just_looking_) {
    if (NGRAPH_VLOG_IS_ON(5)) {
      NGRAPH_VLOG(5) << "Variable " << ctx->op_kernel().name() << ": marking "
                     << DMAHelper::base(var->tensor());
    }
    tracker_->MarkStale(DMAHelper::base(var->tensor()));
    if (NGRAPH_VLOG_IS_ON(5)) {
      NGRAPH_VLOG(5) << "Variable " << ctx->op_kernel().name() << ": marked "
                     << DMAHelper::base(var->tensor());
    }
  }
  // To output a reference.  Caller retains ownership of mu and tensor_for_ref,
  // and they must outlive all uses within the step. See comment above.
  // REQUIRES: IsRefType(expected_output_dtype(index))
  ctx->set_output_ref(0, var->mu(), var->tensor());

  if (ctx->track_allocations() && var->tensor()->IsInitialized()) {
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);
    ctx->record_persistent_memory_allocation(var->tensor()->AllocatedBytes());
  }
  var->Unref();
}

REGISTER_OP("NGraphVariable") 
    .Output("ref: Ref(dtype)")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("just_looking: bool = false")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);

REGISTER_KERNEL_BUILDER(Name("NGraphVariable").Device(DEVICE_CPU),
                        NGraphVariableOp);

}  // namespace ngraph_bridge

}  // namespace tensorflow
