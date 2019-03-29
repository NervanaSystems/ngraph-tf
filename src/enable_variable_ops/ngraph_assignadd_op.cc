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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/default/logging.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "ngraph/runtime/backend.hpp"
#include "ngraph_backend_manager.h"
#include "ngraph_catalog.h"
#include "ngraph_freshness_tracker.h"
#include "ngraph_timer.h"
#include "ngraph_utils.h"
#include "ngraph_var.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

/* -------------------------------------------------
//
// NGraphAssignAddOp
//
---------------------------------------------------*/

// Computes *input[0] = *input[0] + input[1]
class NGraphAssignAddOp : public OpKernel {
 public:
  explicit NGraphAssignAddOp(OpKernelConstruction* context)
      : OpKernel(context), just_looking_(false), copy_to_tf_(false) {
    OP_REQUIRES_OK(context, context->GetAttr("just_looking", &just_looking_));
    OP_REQUIRES_OK(context, context->GetAttr("copy_to_tf", &copy_to_tf_));
    OP_REQUIRES_OK(context, context->GetAttr("ngraph_graph_id", &ng_graph_id_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("_ngraph_backend", &ng_backend_name_));
    NGRAPH_VLOG(4) << "NGraphAssignAdd:: Constructor called for: "
                   << def().name() << " ,just looking " << just_looking_
                   << " ,copy-to-tf " << copy_to_tf_ << " ,Graph ID "
                   << ng_graph_id_;

    OP_REQUIRES(context, IsRefType(context->input_type(0)),
                errors::InvalidArgument("lhs input needs to be a ref type"));
    my_instance_id = s_instance_count;
    s_instance_count++;
  }

  void Compute(OpKernelContext* context) override {
    std::ostringstream oss;
    oss << "Execute: AssignAdd_" << my_instance_id << ": " << name();
    Event event_compute(oss.str().c_str(), name().c_str());

    NGRAPH_VLOG(4) << "NGraphAssignAdd:: Compute called for: " << def().name()
                   << " ,just looking " << just_looking_ << " ,copy-to-tf "
                   << copy_to_tf_ << " ,Graph ID " << ng_graph_id_;

    bool log_copies = false;
    OP_REQUIRES_OK(context, IsCopyLogEnabled(ng_graph_id_, log_copies));
    std::stringstream copy_log_str;
    copy_log_str << "KERNEL[" << type_string() << "]: " << name()
                 << " ,Copy_TF " << PrintBool(copy_to_tf_) << " ,Just_Looking "
                 << PrintBool(just_looking_) << "\n";
    int number_of_copies = 0;

    bool ref_exists =
        NGraphCatalog::ExistsInCatalog(ng_graph_id_, def().name(), 0);
    if (!ref_exists) {
      OP_REQUIRES(
          context, ref_exists,
          errors::Internal(
              "Caught exception : RefInput to NGAssignAdd not found \n"));
    }

    Event event_lookup("NGAssignAdd: Lookup input reference", name().c_str());
    string get_ref_var_name =
        NGraphCatalog::GetInputSharedName(ng_graph_id_, def().name(), 0);

    NGraphVar* var;
    OP_REQUIRES_OK(context,
                   context->resource_manager()->Lookup<NGraphVar>(
                       context->resource_manager()->default_container(),
                       get_ref_var_name, &var));

    event_lookup.Stop();
    Event::WriteTrace(event_lookup);

    // Sync before computing the value
    if (var->need_sync_ng_tensor()) {
      number_of_copies++;
      copy_log_str << "Var_Sync ";
      Event sync_ng_tensor("NGAssignAdd: sync_ng_tensor", name().c_str());

      NGRAPH_VLOG(4)
          << "In AssignAdd, ng tensor behind, needs to sync with tf-tensor";
      WriteNGTensor(var->ng_tensor(), var->tensor());
      // TODO: Is it safe to set sync as false after this sync
      var->sync_ng_tensor(false);
      sync_ng_tensor.Stop();
      Event::WriteTrace(sync_ng_tensor);
    }

    // get the nGraphTensor Variable
    shared_ptr<ngraph::runtime::Tensor> ng_tensor_to_assign = var->ng_tensor();

    // Create Backend
    BackendManager::CreateBackend(ng_backend_name_);
    ng::runtime::Backend* op_backend =
        BackendManager::GetBackend(ng_backend_name_);

    // Get the Value to be added
    const Tensor& rhs = context->input(1);
    // We always return the input ref.
    context->forward_ref_input_to_ref_output(0, 0);

    string valkey = to_string(ng_graph_id_) + "_" + def().input(1);
    bool valref_exists = NGraphCatalog::ExistsInOutputCatalog(valkey);
    shared_ptr<ng::runtime::Tensor> ng_val;

    if (valref_exists) {
      // Value is from encap
      Event input_from_catelog("NGAssignAdd: Getting input from catalog",
                               name().c_str());

      NGRAPH_VLOG(4) << "NGraphAssignAdd::Getting from catalog : " << valkey;
      ng_val = NGraphCatalog::GetNgTensorFromOutputCatalog(valkey);
      input_from_catelog.Stop();

      Event::WriteTrace(input_from_catelog);
    } else {
      number_of_copies++;
      copy_log_str << " COPY_INP_VAL[0]";
      Event input_from_TF("NGAssignAdd: Getting input from TF", name().c_str());

      NGRAPH_VLOG(4) << "NGraphAssignAdd::Getting from TF : " << valkey;
      TensorShape tfshape = rhs.shape();
      ng::Shape ng_shape(tfshape.dims());
      for (int j = 0; j < tfshape.dims(); ++j) {
        ng_shape[j] = tfshape.dim_size(j);
      }
      ng::element::Type ng_element_type;
      OP_REQUIRES_OK(context, TFDataTypeToNGraphElementType(rhs.dtype(),
                                                            &ng_element_type));
      void* tf_src_ptr = (void*)DMAHelper::base(&rhs);
      ng_val = op_backend->create_tensor(ng_element_type, ng_shape);
      ng_val->write(tf_src_ptr, 0, ng_val->get_element_count() *
                                       ng_val->get_element_type().size());
      Event::WriteTrace(input_from_TF);
    }

    // Create nGraph Function
    // Create Input Tensor Vector
    vector<shared_ptr<ng::runtime::Tensor>> ng_inputs = {ng_tensor_to_assign,
                                                         ng_val};
    NGRAPH_VLOG(4) << " Input Tensors Created ";

    std::stringstream signature_ss;
    for (int i = 0; i < ng_inputs.size(); i++) {
      auto ngt = ng_inputs[i];
      for (const auto& x : ng_inputs[i]->get_shape()) {
        signature_ss << x << ",";
      }
      signature_ss << ";";
    }

    signature_ss << "/";
    std::string signature = signature_ss.str();
    NGRAPH_VLOG(4) << " Signature " << signature;

    if (ng_exec_map.find(signature) == ng_exec_map.end()) {
      Event cache_miss("NGAssignAdd: Cache miss, compiling", name().c_str());

      // create and compile function
      NGRAPH_VLOG(4) << " Cache miss ";
      auto V = make_shared<ng::op::Parameter>(
          ng_tensor_to_assign->get_element_type(),
          ng_tensor_to_assign->get_shape());
      auto Val = make_shared<ng::op::Parameter>(ng_val->get_element_type(),
                                                ng_val->get_shape());
      auto add = make_shared<ng::op::Add>(V, Val);

      auto ng_function = make_shared<ng::Function>(ng::NodeVector{add},
                                                   ng::ParameterVector{V, Val});

      NGRAPH_VLOG(4) << " Created Function ";

      // Compile Function to get executable
      auto ng_exec_temp = op_backend->compile(ng_function);
      NGRAPH_VLOG(4) << " Compiled Function ";
      ng_exec_map[signature] = ng_exec_temp;
      cache_miss.Stop();
      Event::WriteTrace(cache_miss);
    }
    auto ng_exec = ng_exec_map[signature];

    // Create Output Tensor Vector
    vector<shared_ptr<ng::runtime::Tensor>> ng_outputs;
    for (int i = 0; i < ng_exec->get_results().size(); i++) {
      auto ng_element = ng_exec->get_results()[i];
      auto ng_shape = ng_element->get_shape();
      auto ng_element_type = ng_element->get_element_type();
      shared_ptr<ng::runtime::Tensor> ng_op =
          op_backend->create_tensor(ng_element_type, ng_shape);
      ng_outputs.push_back(ng_op);
    }
    NGRAPH_VLOG(4) << " Output Tensors Created ";

    // Call Executable
    Event call_executable("NGAssignAdd: Calling executable", name().c_str());
    ng_exec->call(ng_outputs, ng_inputs);
    NGRAPH_VLOG(4) << " Call Executed ";
    call_executable.Stop();
    Event::WriteTrace(call_executable);

    // Assign to the variable
    Event result_copy("NGAssignAdd: Copy executed output to variable tensor",
                      name().c_str());
    ng_tensor_to_assign->copy_from(*(ng_outputs[0]));
    result_copy.Stop();
    Event::WriteTrace(result_copy);

    mutex_lock l(*context->input_ref_mutex(0));
    Tensor old_lhs = context->mutable_input(0, /* lock_held */ true);
    auto tf_tensor = var->tensor();

    if (copy_to_tf_) {
      number_of_copies++;
      copy_log_str << " COPY_TF ";
      Event copy_to_tf("NGAssignAdd: Copy to TF tensor", name().c_str());
      ReadNGTensor(ng_tensor_to_assign, &old_lhs);

      copy_to_tf.Stop();
      Event::WriteTrace(copy_to_tf);

      if (just_looking_) {
        // Some tf op will just use the val

      } else {
        // Some tf op might update the ng-tensor value so mark it stale
        copy_log_str << " SET_SYNC ";
        var->sync_ng_tensor(true);
      }
    }

    copy_log_str << " Number of copies " << number_of_copies << "\n";
    if (log_copies) {
      cout << copy_log_str.str();
    }

    // Unref Var
    var->Unref();
    event_compute.Stop();
    Event::WriteTrace(event_compute);
  }

 private:
  bool just_looking_;
  bool copy_to_tf_;
  int ng_graph_id_;
  string ng_backend_name_;
  std::unordered_map<std::string, std::shared_ptr<ngraph::runtime::Executable>>
      ng_exec_map;
  static int s_instance_count;
  int my_instance_id{0};

  // bool use_exclusive_lock_; //TF op has this
  ~NGraphAssignAddOp() override {
    // Release the backend
    BackendManager::ReleaseBackend(ng_backend_name_);
    NGRAPH_VLOG(2) << "~NGraphAssignAddOp";
  }
};

int NGraphAssignAddOp::s_instance_count = 0;

REGISTER_OP("NGraphAssignAdd")
    .Input("ref: Ref(T)")
    .Input("value: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: type")
    .Attr("validate_shape: bool = true")
    .Attr("use_locking: bool = true")
    .Attr("just_looking: bool = false")
    .Attr("copy_to_tf: bool = false")
    .Attr("ngraph_graph_id: int");

REGISTER_KERNEL_BUILDER(Name("NGraphAssignAdd").Device(DEVICE_CPU),
                        NGraphAssignAddOp);

}  // namespace ngraph_bridge

}  // namespace tensorflow
