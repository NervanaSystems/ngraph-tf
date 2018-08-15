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
#include "ngraph_mark_for_clustering.h"
#include "ngraph_utils.h"

#include "ngraph/runtime/interpreter/int_backend.hpp"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

REGISTER_OP("NGraphEncapsulate")
    .Input("args: Targuments")
    .Attr("Targuments: list(type) >= 0")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .Attr("ngraph_cluster: int")
    .SetIsStateful()
    .Doc("nGraph Encapsulation Op. For use by the nGraph JIT only.");

class NGraphEncapsulateOp : public OpKernel {
 public:
  explicit NGraphEncapsulateOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), m_graph(OpRegistry::Global()), m_freshness_tracker(nullptr) {
    GraphDef* graph_def;

    OP_REQUIRES_OK(ctx, ctx->GetAttr<int>("ngraph_cluster", &m_ngraph_cluster));
    graph_def = NGraphClusterManager::GetClusterGraph(m_ngraph_cluster);

    GraphConstructorOptions opts;
    opts.allow_internal_ops = true;
    OP_REQUIRES_OK(ctx, ConvertGraphDefToGraph(opts, *graph_def, &m_graph));

    //
    // Initialize the "m_input_is_static" vector as follows:
    // (1) create m_input_is_static with n+1 elements, where n is the max arg
    //     index
    // (2) for each _Arg node n, set m_input_is_static[n.index] to true if n
    //     is driving any static input; else set it to false.
    //

    // Create the vector.
    int32 max_arg_index = -1;
    std::vector<const Node*> arg_nodes;

    for (auto node : m_graph.nodes()) {
      if (node->type_string() == "_Arg") {
        arg_nodes.push_back(node);

        int32 index;
        OP_REQUIRES_OK(ctx, GetNodeAttr(node->attrs(),"index",&index));
        if (index > max_arg_index) max_arg_index = index;
      }
    }

    m_input_is_static = std::vector<bool>(max_arg_index+1,false);

    // Fill the vector.
    for (auto node : arg_nodes) {
      int32 index;
      OP_REQUIRES_OK(ctx, GetNodeAttr(node->attrs(),"index",&index));

      bool is_static = false;
      for (auto edge : node->out_edges()) {
        if (edge->IsControlEdge() || !edge->dst()->IsOp()) {
          continue;
        }

        NGRAPH_VLOG(5) << "For arg " << index << " checking edge " << edge->DebugString();

        if (InputIsStatic(edge->dst(), edge->dst_input())) {
          NGRAPH_VLOG(5) << "Marking edge static: " << edge->DebugString();
          is_static = true;
          break;
        }
      }

      NGRAPH_VLOG(5) << "Marking arg " << index << " is_static: " << is_static;
      m_input_is_static[index] = is_static;
    }

    // Create the backend
    if (m_ng_backend == nullptr) {
#if defined(NGRAPH_EMBEDDED_IN_TENSORFLOW)
      m_ng_backend = std::make_shared<ng::runtime::interpreter::INTBackend>();
#else
      m_ng_backend = ng::runtime::Backend::create("CPU");
#endif
      OP_REQUIRES(ctx, m_ng_backend != nullptr,
                  errors::InvalidArgument("Cannot create nGraph backend"));
    }
  }

  ~NGraphEncapsulateOp() override {
    // If the kernel goes away, we must de-register all of its cached functions
    // from the freshness tracker.
    if (m_freshness_tracker != nullptr) {
      for (auto kv : m_ng_functions) {
        m_freshness_tracker->RemoveUser(kv.second);
      }

      // TODO(amprocte): We should be able to unref the tracker here, but it
      // seems to screw things up in the C++ unit tests.
      // m_freshness_tracker->Unref();
    }
  }

  template<typename T>
  static void TensorDataToStream(std::ostream& ostream, int64 n_elements, const char* data) {
    const T* data_T = reinterpret_cast<const T*>(data);
    for (size_t i = 0; i < n_elements; i++) {
      ostream << data_T[i] << ",";
    }
  }

  static Status TensorToStream(std::ostream& ostream, const Tensor& tensor) {
    const char* data = tensor.tensor_data().data();
    int64 n_elements = tensor.NumElements();
    switch(tensor.dtype()) {
      case DT_HALF:
        TensorDataToStream<Eigen::half>(ostream, n_elements, data);
        break;
      case DT_FLOAT:
        TensorDataToStream<float>(ostream, n_elements, data);
        break;
      case DT_DOUBLE:
        TensorDataToStream<double>(ostream, n_elements, data);
        break;
      case DT_UINT32:
        TensorDataToStream<uint32>(ostream, n_elements, data);
        break;
      case DT_INT32:
        TensorDataToStream<int32>(ostream, n_elements, data);
        break;
      case DT_UINT8:
      case DT_QUINT8:
        TensorDataToStream<uint8>(ostream, n_elements, data);
        break;
      case DT_UINT16:
      case DT_QUINT16:
        TensorDataToStream<uint16>(ostream, n_elements, data);
        break;
      case DT_INT8:
      case DT_QINT8:
        TensorDataToStream<int8>(ostream, n_elements, data);
        break;
      case DT_INT16:
      case DT_QINT16:
        TensorDataToStream<int16>(ostream, n_elements, data);
        break;
      case DT_UINT64:
        TensorDataToStream<uint64>(ostream, n_elements, data);
        break;
      case DT_INT64:
        TensorDataToStream<int64>(ostream, n_elements, data);
        break;
      case DT_BOOL:
        TensorDataToStream<bool>(ostream, n_elements, data);
        break;
      default:
        return errors::Internal("TensorToStream got unsupported data type ", DataType_Name(tensor.dtype()));
        break;
    }
    return Status::OK();
  }

  // TODO(amprocte): this needs to be made thread-safe (compilation cache OK?).
  void Compute(OpKernelContext* ctx) override {
    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute starting for cluster " << m_ngraph_cluster;

    // Get the inputs
    std::vector<TensorShape> input_shapes;
    std::stringstream signature_ss;
    for (int i = 0; i < ctx->num_inputs(); i++) {
      const Tensor& input_tensor = ctx->input(i);
      input_shapes.push_back(input_tensor.shape());
      for (const auto& x : input_tensor.shape()) {
        signature_ss << x.size << ",";
      }
      signature_ss << ";";
    }

    signature_ss << "/";

    std::vector<const Tensor*> static_input_map(ctx->num_inputs());
    for (int i = 0; i < ctx->num_inputs(); i++) {
      const Tensor& input_tensor = ctx->input(i);
      if (m_input_is_static[i]) {
        static_input_map[i] = &input_tensor;
        OP_REQUIRES_OK(ctx, TensorToStream(signature_ss, input_tensor));
        signature_ss << ";";
      }
    }

    std::shared_ptr<ngraph::Function> ng_function;
    std::string signature = signature_ss.str();

    if (NGRAPH_VLOG_IS_ON(5)) {
      NGRAPH_VLOG(5) << "Computed signature: " << signature;
    }

    auto it = m_ng_functions.find(signature);

    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute got inputs for cluster "
                   << m_ngraph_cluster;

    // Compile the graph using nGraph.
    //
    // TODO(amprocte): Investigate performance of the compilation cache.
    if (it == m_ng_functions.end()) {
      NGRAPH_VLOG(1) << "Compilation cache miss: " << ctx->op_kernel().name();
      OP_REQUIRES_OK(
          ctx, Builder::TranslateGraph(input_shapes, static_input_map, &m_graph, ng_function));

      // Serialize to nGraph if needed
      if (std::getenv("NGRAPH_ENABLE_SERIALIZE") != nullptr) {
        std::string file_name =
            "tf_function_" + ctx->op_kernel().name() + ".json";
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

    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute got graph for cluster "
                   << m_ngraph_cluster;

    if (m_freshness_tracker == nullptr) {
      auto creator = [](NGraphFreshnessTracker** tracker) {
        *tracker = new NGraphFreshnessTracker();
        return Status::OK();
      };
      OP_REQUIRES_OK(
          ctx,
          ctx->resource_manager()->LookupOrCreate<NGraphFreshnessTracker>(
              ctx->resource_manager()->default_container(),
              "ngraph_freshness_tracker", &m_freshness_tracker, creator));
    }

    NGRAPH_VLOG(4)
        << "NGraphEncapsulateOp::Compute got freshness tracker for cluster "
        << m_ngraph_cluster;

    // Allocate tensors for arguments.
    vector<shared_ptr<ng::runtime::TensorView>> ng_inputs;

    auto& last_used_src_ptrs = m_last_used_src_ptrs_map[ng_function];
    last_used_src_ptrs.resize(input_shapes.size());

    for (int i = 0; i < input_shapes.size(); i++) {
      ng::Shape ng_shape(input_shapes[i].dims());
      for (int j = 0; j < input_shapes[i].dims(); ++j) {
        ng_shape[j] = input_shapes[i].dim_size(j);
      }

      ng::element::Type ng_element_type;
      OP_REQUIRES_OK(ctx, TFDataTypeToNGraphElementType(ctx->input(i).dtype(),
                                                        &ng_element_type));

      void* src_ptr = (void*)DMAHelper::base(&ctx->input(i));
      auto t = m_ng_backend->create_tensor(ng_element_type, ng_shape, src_ptr);

      // Mark each tensor as non-stale if:
      //
      //   1. the freshness tracker says the tensor has not changed since
      //      the last time ng_function was called, and
      //   2. we are using the same tensor in this argument position as
      //      the one we used last time ng_function was called.
      if (m_freshness_tracker->IsFresh(src_ptr, ng_function) &&
          src_ptr == last_used_src_ptrs[i]) {
        NGRAPH_VLOG(5) << "input " << i << " at " << src_ptr << ": not stale";
        t->set_stale(false);
      } else {
        NGRAPH_VLOG(5) << "input " << i << " at " << src_ptr << ": stale";
        t->set_stale(true);
      }
      last_used_src_ptrs[i] = src_ptr;
      ng_inputs.push_back(t);
    }

    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute allocated argument tensors "
                      "for cluster "
                   << m_ngraph_cluster;

    // Allocate tensors for the results.
    vector<shared_ptr<ng::runtime::TensorView>> outputs;
    for (auto i = 0; i < ng_function->get_output_size(); i++) {
      auto shape = ng_function->get_output_shape(i);
      auto elem_type = ng_function->get_output_element_type(i);

      // Create the TF output tensor
      vector<int64> dims;
      for (auto dim : shape) {
        dims.push_back(dim);
      }
      TensorShape tf_shape(dims);
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, tf_shape, &output_tensor));

      // Make sure the nGraph-inferred element type agrees with what TensorFlow
      // expected.
      ng::element::Type expected_elem_type;
      OP_REQUIRES_OK(
          ctx, TFDataTypeToNGraphElementType(ctx->expected_output_dtype(i),
                                             &expected_elem_type));
      OP_REQUIRES(
          ctx, elem_type == expected_elem_type,
          errors::Internal("Element type inferred by nGraph does not match "
                           "the element type expected by TensorFlow"));

      // Create the nGraph output tensor
      void* dst_ptr = DMAHelper::base(output_tensor);
      auto t_result = m_ng_backend->create_tensor(elem_type, shape, dst_ptr);

      outputs.push_back(t_result);
    }

    NGRAPH_VLOG(4)
        << "NGraphEncapsulateOp::Compute allocated result tensors for cluster "
        << m_ngraph_cluster;

    // Execute the nGraph function.
    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute call starting for cluster "
                   << m_ngraph_cluster;
    m_ng_backend->call(ng_function, outputs, ng_inputs);
    NGRAPH_VLOG(4) << "NGraphEncapsulateOp::Compute call done for cluster " << m_ngraph_cluster;

    // Mark input tensors as fresh for the next time around.
    for (int i = 0; i < input_shapes.size(); i++) {
      void* src_ptr = (void*)DMAHelper::base(&ctx->input(i));
      m_freshness_tracker->MarkFresh(src_ptr, ng_function);
    }

    NGRAPH_VLOG(4)
        << "NGraphEncapsulateOp::Compute done marking fresh for cluster "
        << m_ngraph_cluster;
  }

 private:
  Graph m_graph;
  std::unordered_map<std::string, std::shared_ptr<ngraph::Function>>
      m_ng_functions;
  std::map<std::shared_ptr<ngraph::Function>, std::vector<const void*>>
      m_last_used_src_ptrs_map;
  NGraphFreshnessTracker* m_freshness_tracker;
  int m_ngraph_cluster;
  static std::shared_ptr<ng::runtime::Backend> m_ng_backend;
  std::vector<bool> m_input_is_static;
};
std::shared_ptr<ng::runtime::Backend> NGraphEncapsulateOp::m_ng_backend;

}  // namespace ngraph_bridge

REGISTER_KERNEL_BUILDER(Name("NGraphEncapsulate").Device(DEVICE_CPU),
                        ngraph_bridge::NGraphEncapsulateOp);

}  // namespace tensorflow
