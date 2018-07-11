/*******************************************************************************
o * Copyright 2017-2018 Intel Corporation
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
#include <fstream>

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"

#include "ngraph/serializer.hpp"

#include "ngraph_builder.h"
#include "ngraph_cluster_manager.h"
#include "ngraph_freshness_tracker.h"
#include "ngraph_log.h"
#include "ngraph_utils.h"

namespace tf = tensorflow;
namespace ngb = ngraph_bridge;

namespace ngraph_bridge {
extern const char* const DEVICE_NGRAPH;

REGISTER_OP("NGraphEncapsulate")
    .Input("args: Targuments")
    .Attr("Targuments: list(type) >= 0")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .Attr("ngraph_cluster: int")
    .SetIsStateful()
    .Doc("nGraph Encapsulation Op. For use by the nGraph JIT only.");

class NGraphEncapsulateOp : public tf::OpKernel {
 public:
  explicit NGraphEncapsulateOp(tf::OpKernelConstruction* ctx)
      : tf::OpKernel(ctx),
        m_graph(tf::OpRegistry::Global()),
        m_freshness_tracker(nullptr) {
    tf::GraphDef* graph_def;

    // TODO(amprocte): need to check status result here.
    OP_REQUIRES_OK(ctx, ctx->GetAttr<int>("ngraph_cluster", &m_ngraph_cluster));
    graph_def = NGraphClusterManager::GetClusterGraph(m_ngraph_cluster);

    tf::GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    // TODO(amprocte): need to check status result here.
    OP_REQUIRES_OK(ctx, tf::ConvertGraphDefToGraph(opts, *graph_def, &m_graph));

    // Create the backend
    if (m_cpu_backend == nullptr) {
      m_cpu_backend = ng::runtime::Backend::create("GPU");
      OP_REQUIRES(ctx, m_cpu_backend != nullptr,
                  tf::errors::InvalidArgument("Cannot create GPU backend"));
    }
  }

  ~NGraphEncapsulateOp() override {
    // If the kernel goes away, we must de-register all of its cached functions
    // from the freshness tracker.
    if (m_freshness_tracker != nullptr) {
      for (auto kv : m_ng_functions) {
        m_freshness_tracker->RemoveUser(kv.second);
      }
    }
    // d-tor
  }

  // TODO(amprocte): this needs to be made thread-safe (compilation cache, and
  // our use of the freshness-tracking stuff probably means we can only execute
  // one instance at a time).
  void Compute(tf::OpKernelContext* ctx) override {
    // Get the inputs
    std::vector<tf::TensorShape> input_shapes;
    std::stringstream signature_ss;
    for (int i = 0; i < ctx->num_inputs(); i++) {
      const tf::Tensor& input_tensor = ctx->input(i);
      input_shapes.push_back(input_tensor.shape());
      for (const auto& x : input_tensor.shape()) {
        signature_ss << x.size << ",";
      }
      signature_ss << ";";
    }

    std::shared_ptr<ngraph::Function> ng_function;
    std::string signature = signature_ss.str();
    auto it = m_ng_functions.find(signature);

    // Compile the graph using nGraph.
    //
    // TODO(amprocte): Investigate performance of the compilation cache.
    if (it == m_ng_functions.end()) {
      OP_REQUIRES_OK(
          ctx, Builder::TranslateGraph(input_shapes, &m_graph, ng_function));

      // Serialize to nGraph if needed
      if (std::getenv("NGRAPH_ENABLE_SERIALIZE") != nullptr) {
        std::string file_name =
            "tf_function_" + ctx->op_kernel().name() + ".js";
        NGRAPH_VLOG(0) << "Serializing graph to: " << file_name;
        std::string js = ngraph::serialize(ng_function, 4);
        {
          std::ofstream f(file_name);
          f << js;
        }
      }

      m_ng_functions[signature] = ng_function;
    } else {
      ng_function = it->second;
    }

    if (m_freshness_tracker == nullptr) {
      auto creator = [this](ngb::NGraphFreshnessTracker** tracker) {
        *tracker = new ngb::NGraphFreshnessTracker();
        return tf::Status::OK();
      };
      OP_REQUIRES_OK(
          ctx,
          ctx->resource_manager()->LookupOrCreate<ngb::NGraphFreshnessTracker>(
              ctx->resource_manager()->default_container(),
              "ngraph_freshness_tracker", &m_freshness_tracker, creator));
    }

    // Allocate tensors for arguments.
    vector<shared_ptr<ng::runtime::TensorView>> ng_inputs;

    auto& last_used_src_ptrs = m_last_used_src_ptrs_map[ng_function];
    last_used_src_ptrs.resize(input_shapes.size());

    for (int i = 0; i < input_shapes.size(); i++) {
      ng::Shape ng_shape(input_shapes[i].dims());
      int input_size = 1;
      for (int j = 0; j < input_shapes[i].dims(); ++j) {
        ng_shape[j] = input_shapes[i].dim_size(j);
        input_size *= ng_shape[j];
      }

      ng::element::Type ng_element_type;
      OP_REQUIRES_OK(ctx, TFDataTypeToNGraphElementType(ctx->input(i).dtype(),
                                                        &ng_element_type));

      void* src_ptr = (void*)tf::DMAHelper::base(&ctx->input(i));
      auto t = m_cpu_backend->create_tensor(ng_element_type, ng_shape);
      t->write(src_ptr, 0, input_size * ng_element_type.size());

      // Mark each tensor as non-stale if:
      //
      //   1. the freshness tracker says the tensor has not changed since
      //      the last time ng_function was called, and
      //   2. we are using the same tensor in this argument position as
      //      the one we used last time ng_function was called.
      if (m_freshness_tracker->IsFresh(src_ptr, ng_function) &&
          src_ptr == last_used_src_ptrs[i]) {
        t->set_stale(false);
      } else {
        t->set_stale(true);
      }
      last_used_src_ptrs[i] = src_ptr;
      ng_inputs.push_back(t);
    }

    // Allocate tensors for the results.
    vector<shared_ptr<ng::runtime::TensorView>> outputs;
    vector<void*> outputs_tf;
    vector<size_t> outputs_size;
    for (auto i = 0; i < ng_function->get_output_size(); i++) {
      auto shape = ng_function->get_output_shape(i);
      auto elem_type = ng_function->get_output_element_type(i);

      // Create the TF output tensor
      vector<tf::int64> dims;
      size_t output_size = 1;
      for (auto dim : shape) {
        output_size *= dim;
        dims.push_back(dim);
      }
      tf::TensorShape tf_shape(dims);
      tf::Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, tf_shape, &output_tensor));

      // Make sure the nGraph-inferred element type agrees with what TensorFlow
      // expected.
      ng::element::Type expected_elem_type;
      OP_REQUIRES_OK(
          ctx, TFDataTypeToNGraphElementType(ctx->expected_output_dtype(i),
                                             &expected_elem_type));
      OP_REQUIRES(
          ctx, elem_type == expected_elem_type,
          tf::errors::Internal("Element type inferred by nGraph does not match "
                               "the element type expected by TensorFlow"));

      // Create the nGraph output tensor
      void* dst_ptr = (void*)tf::DMAHelper::base(output_tensor);
      outputs_tf.push_back(dst_ptr);
      outputs_size.push_back(output_size * elem_type.size());
      auto t_result = m_cpu_backend->create_tensor(elem_type, shape);

      outputs.push_back(t_result);
    }

    // Execute the nGraph function.
    NGRAPH_VLOG(4) << "call starting for cluster " << m_ngraph_cluster;
    m_cpu_backend->call(ng_function, outputs, ng_inputs);
    NGRAPH_VLOG(4) << "call done for cluster " << m_ngraph_cluster;
    
    for(int i = 0; i < outputs.size(); i++)
    {
       outputs[i]->read(outputs_tf[i], 0, outputs_size[i]);
    } 

    // Mark input tensors as fresh for the next time around.
    for (int i = 0; i < input_shapes.size(); i++) {
      void* src_ptr = (void*)tf::DMAHelper::base(&ctx->input(i));
      m_freshness_tracker->MarkFresh(src_ptr, ng_function);
    }
  }

 private:
  tf::Graph m_graph;
  std::unordered_map<std::string, std::shared_ptr<ngraph::Function>>
      m_ng_functions;
  std::map<std::shared_ptr<ngraph::Function>, std::vector<const void*>>
      m_last_used_src_ptrs_map;
  ngb::NGraphFreshnessTracker* m_freshness_tracker;
  int m_ngraph_cluster;
  static std::shared_ptr<ng::runtime::Backend> m_cpu_backend;
};
std::shared_ptr<ng::runtime::Backend> NGraphEncapsulateOp::m_cpu_backend;

}  // namespace ngraph_bridge

namespace tensorflow {
REGISTER_KERNEL_BUILDER(
    Name("NGraphEncapsulate").Device(ngraph_bridge::DEVICE_NGRAPH),
    ngraph_bridge::NGraphEncapsulateOp);
}
