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
#include "ngraph/shape.hpp"
#include "ngraph_backend_manager.h"
#include "ngraph_catalog.h"
#include "ngraph_freshness_tracker.h"
#include "ngraph_utils.h"
#include "ngraph_var.h"
#include "ngraph_timer.h"

using namespace std;
// using namespace ngraph;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

/* -------------------------------------------------
//
// NGraphApplyGradientDescentOp
//
---------------------------------------------------*/

class NGraphApplyGradientDescentOp : public OpKernel {
 private:
  bool just_looking_;
  bool copy_to_tf_;
  int ng_graph_id_;
  string ng_backend_name_;
  std::unordered_map<std::string, std::shared_ptr<ngraph::runtime::Executable>>
      ng_exec_map;
  int my_instance_id{0};
  static int s_instance_count;

 public:
  explicit NGraphApplyGradientDescentOp(OpKernelConstruction* context)
      : OpKernel(context), just_looking_(false), copy_to_tf_(false) {
    OP_REQUIRES_OK(context, context->GetAttr("just_looking", &just_looking_));
    OP_REQUIRES_OK(context, context->GetAttr("copy_to_tf", &copy_to_tf_));
    OP_REQUIRES_OK(context, context->GetAttr("ngraph_graph_id", &ng_graph_id_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("_ngraph_backend", &ng_backend_name_));

    OP_REQUIRES(context, IsRefType(context->input_type(0)),
                errors::InvalidArgument("The first input must be a ref type "
                                        "for NGraphApplyGraidenteDescent"));

    NGRAPH_VLOG(4) << "NGraphApplyGradientDescent:: Constructor called for: " << def().name()
                   << " ,just looking " << just_looking_ << " ,copy-to-tf "
                   << copy_to_tf_ <<" ,Graph ID "<<ng_graph_id_;


    my_instance_id = s_instance_count;
    s_instance_count++;
  }

  //---------------------------------------------------------------------------
  //  ~NGraphApplyGradientDescentOp()
  //---------------------------------------------------------------------------
  ~NGraphApplyGradientDescentOp() override {
    // Release the backend
    BackendManager::ReleaseBackend(ng_backend_name_);
    NGRAPH_VLOG(2) << " ~NGraphApplyGradientDescentOp";
  }

  void Compute(OpKernelContext* context) override {
    std::ostringstream oss;
    oss << "Execute: NGApplyGradientDescent compute" << my_instance_id << ": " << name();
    Event event_compute(oss.str().c_str(), name().c_str());

    NGRAPH_VLOG(4) << "NGraphApplyGradientDescent:: Compute called for: " << def().name()
                   << " ,just looking " << just_looking_ << " ,copy-to-tf "
                   << copy_to_tf_ <<" ,Graph ID "<<ng_graph_id_;

    bool log_copies = false;
    OP_REQUIRES_OK(context ,IsCopyLogEnabled(ng_graph_id_, log_copies));
    std::stringstream copy_log_str;
    copy_log_str<<"KERNEL["<< type_string() <<"]: " << name() << " ,Copy_TF "<< PrintBool(copy_to_tf_) <<" ,Just_Looking "<< PrintBool(just_looking_)<<"\n";
    int number_of_copies = 0;

    // Get the 1st input ref from input_catelog (NGraphVar ->
    // NGraphApplyGraidentDescent)
    bool ref_exists =
        NGraphCatalog::ExistsInCatalog(ng_graph_id_, def().name(), 0);
    if (!ref_exists) {
      OP_REQUIRES(context, ref_exists,
                  errors::Internal("Caught exception : RefInput to "
                                   "NGraphApplyGradientDescent not found in Catalog \n"));
    }
    string get_ref_var_name =
        NGraphCatalog::GetInputSharedName(ng_graph_id_, def().name(), 0);
    
    //Throw error
    NGraphVar* var;
    if (context->resource_manager()->Lookup<NGraphVar>(
            context->resource_manager()->default_container(), get_ref_var_name,
            &var) == Status::OK()) {
      NGRAPH_VLOG(4) << "NGraphApplyGradientDescent:: Found variable in resource manager";
    } else {
      NGRAPH_VLOG(4) << "NGraphApplyGradientDescent:: Not Found var in resource manager";
    }

    // Sync before using the variable for computation
    if (var->need_sync_ng_tensor()) {
      number_of_copies++;
      copy_log_str<<"Var_Sync ";
      NGRAPH_VLOG(4) << "in ApplyGradientDescent, ng tensor behind, needs to "
                        "sync with tf-tensor";
      WriteNGTensor(var->ng_tensor(), var->tensor());
      var->sync_ng_tensor(false);
    }

    // get the nGraphTensor
    shared_ptr<ngraph::runtime::Tensor> ng_tensor_to_assign = var->ng_tensor();

    // Construct the ngraph graph for gradient descent formula
    unordered_map<int, shared_ptr<ng::runtime::Tensor>> input_to_ng_tensor_map;

    // Create Backend
    BackendManager::CreateBackend(ng_backend_name_);
    ng::runtime::Backend* op_backend =
        BackendManager::GetBackend(ng_backend_name_);

    // Check for the two other inputs for NGraphApplyGraidenetDescent
    // if they are coming from NGraphEncapsulates
    for (int i = 1; i < 3; i++) {
      string valkey = to_string(ng_graph_id_) + "_" + def().input(i);
      NGRAPH_VLOG(4) << "NGraphAGD input " << i << " with key " << valkey;

      // checking in output catelog
      bool valref_exists = NGraphCatalog::ExistsInOutputCatalog(valkey);

      if (valref_exists) {
        // Value is from encap
        NGRAPH_VLOG(4) << "NGraphApplyGradientDescent::Getting from Catalog : " << valkey;
        auto ng_val = NGraphCatalog::GetNgTensorFromOutputCatalog(valkey);
        input_to_ng_tensor_map[i] = ng_val;
        NGRAPH_VLOG(4) << "Insert ng_tensor input " << i
                       << "in input_to_ng_tensor_map ";
      } else {
         number_of_copies++;
        copy_log_str<<" COPY_INP_VAL["<<i<<"]";
        NGRAPH_VLOG(4) << "NGraphApplyGradientDescent::Getting from TF : " << valkey;
        const Tensor& rhs = context->input(i);
        void* tf_src_ptr = (void*)DMAHelper::base(&rhs);

        // TF datatype to nGraph element type
        DataType dtype = rhs.dtype();
        ng::element::Type ng_element_type;
        TFDataTypeToNGraphElementType(dtype, &ng_element_type);

        TensorShape shape = rhs.shape();
        // TF TensorShape to nGraphShape
        ng::Shape ng_shape(shape.dims());
        for (int j = 0; j < shape.dims(); ++j) {
          ng_shape[j] = shape.dim_size(j);
        }

        // Create nGTensor
        auto ng_tensor = op_backend->create_tensor(ng_element_type, ng_shape);
        ng_tensor->write(tf_src_ptr, 0,
                         ng_tensor->get_element_count() *
                             ng_tensor->get_element_type().size());
        input_to_ng_tensor_map[i] = ng_tensor;
      }
    }  // end of getting inputs for NGraphApplyGradientDescent

    NGRAPH_VLOG(4) << "Size should be 2 " << input_to_ng_tensor_map.size();
    if (input_to_ng_tensor_map.size() < 2) {
      OP_REQUIRES(context, input_to_ng_tensor_map.size(),
                  errors::Internal("Caught exception : Missing inputs to  "
                                   "NGraphApplyGradientDescent \n"));
    }

    // Create Input Tensor Vector
    vector<shared_ptr<ng::runtime::Tensor>> ng_inputs = {
        ng_tensor_to_assign, input_to_ng_tensor_map[1],
        input_to_ng_tensor_map[2]};

    // Compute the function signature as key
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
      // create and compile function
      NGRAPH_VLOG(4) << " Cache miss in NGraphApplyGraidentDescent ";

      // Build the graph for var - (alpha * delta)
      auto var_param = std::make_shared<ng::op::Parameter>(
          ng_tensor_to_assign->get_element_type(),
          ng_tensor_to_assign->get_shape());
      auto alpha_param = std::make_shared<ng::op::Parameter>(
          input_to_ng_tensor_map[1]->get_element_type(),
          input_to_ng_tensor_map[1]->get_shape());
      auto delta_param = std::make_shared<ng::op::Parameter>(
          input_to_ng_tensor_map[2]->get_element_type(),
          input_to_ng_tensor_map[2]->get_shape());
      NGRAPH_VLOG(4) << "Constructed the parameters for the graph";

      std::shared_ptr<ng::Node> ng_alpha_param, ng_delta_param;
      std::tie(ng_alpha_param, ng_delta_param) = ng::builder::numpy_broadcast(
          std::make_pair(alpha_param, delta_param));

      auto t0 =
          std::make_shared<ng::op::Multiply>(ng_alpha_param, ng_delta_param);
      auto t1 = std::make_shared<ng::op::Subtract>(var_param, t0);

      auto ng_function = std::make_shared<ng::Function>(
          ng::NodeVector{t1},
          ng::ParameterVector{var_param, alpha_param, delta_param});
      NGRAPH_VLOG(4) << "Constructed ApplyGradientDescent ng_function ";

      // Compile Function to get executable
      auto ng_exec_temp = op_backend->compile(ng_function);
      NGRAPH_VLOG(4) << "Compiled ApplyGradientDescent ng_function ";
      ng_exec_map[signature] = ng_exec_temp;  // cache the ng_executable
    }
    auto ng_exec = ng_exec_map[signature];

    // Create Output Tensor Vector
    std::vector<shared_ptr<ng::runtime::Tensor>> ng_outputs;
    for (auto i = 0; i < ng_exec->get_results().size(); i++) {
      auto ng_element = ng_exec->get_results()[i];
      auto ng_shape = ng_element->get_shape();
      auto ng_element_type = ng_element->get_element_type();

      auto ng_op = op_backend->create_tensor(ng_element_type, ng_shape);
      ng_outputs.push_back(ng_op);
    }

    // Call Executable
    ng_exec->call(ng_outputs, ng_inputs);
    NGRAPH_VLOG(4) << "Finished calling the compiled executable ";

    // Assign to the variable
    ng_tensor_to_assign->copy_from(*ng_outputs[0]);
    
    // Set the output
    context->forward_ref_input_to_ref_output(0, 0);

    mutex_lock l(*context->input_ref_mutex(0));
    Tensor old_lhs = context->mutable_input(0, /* lock_held */ true);

    // Update the tf tensor alsoe
    if (copy_to_tf_) {
      number_of_copies++;
      copy_log_str<<" COPY_TF ";
      ReadNGTensor(ng_tensor_to_assign, &old_lhs);
      NGRAPH_VLOG(4) << "Copying to TF Tensor";

      if (just_looking_) {
        // Some tf op will just use the val

      } else {
        copy_log_str<<" SET_SYNC ";
        // Some tf op might update the ng-tensor value so mark it stale
        var->sync_ng_tensor(true);
      }
    }

    copy_log_str<<" Number of copies "<<number_of_copies<<"\n";
    if(log_copies){ 
      cout<< copy_log_str.str();
    }

    // Unref Var
    var->Unref();
    event_compute.Stop();
    Event::WriteTrace(event_compute);
  }  // end of compute function
};   // end of NGraphApplyGradientDescent class definition

int NGraphApplyGradientDescentOp::s_instance_count = 0;

REGISTER_OP("NGraphApplyGradientDescent")
    .Input("var: Ref(T)")
    .Input("alpha: T")
    .Input("delta: T")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("just_looking: bool = false")
    .Attr("copy_to_tf: bool = false")
    .Attr("ngraph_graph_id: int");

REGISTER_KERNEL_BUILDER(Name("NGraphApplyGradientDescent").Device(DEVICE_CPU),
                        NGraphApplyGradientDescentOp);

}  // namespace ngraph_bridge

}  // namespace tensorflow
