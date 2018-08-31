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

#include "ngraph_builder1.h"
#include "ngraph_conversions.h"
#include "ngraph_utils.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

// TODO (sarkars): combine the 2 MakePaddings together? with default args?
// would want the default args to be at the end of the paramlist,
// but 'outputs' (like ng_padding_below, ng_padding_above) are usually at the
// end of the param list
// TODO (sarkars): why does MakePadding handle only 2 dimensions... generalize
// it?
// TODO (sarkars): add unit tests for 1D conv. 1d pooling etc. check if
// MakePadding works in that case
template <typename T>
void MakePadding(const std::string& tf_padding_type,
                 const ngraph::Shape& ng_image_shape,
                 const ngraph::Shape& ng_kernel_shape,
                 const ngraph::Strides& ng_strides,
                 const ngraph::Shape& ng_dilations, T& ng_padding_below,
                 T& ng_padding_above) {
  ngraph::Shape ng_dilation_kernel_shape{
      (ng_kernel_shape[0] - 1) * ng_dilations[0] + 1,
      (ng_kernel_shape[1] - 1) * ng_dilations[1] + 1};

  MakePadding(tf_padding_type, ng_image_shape, ng_dilation_kernel_shape,
              ng_strides, ng_padding_below, ng_padding_above);
}

template <typename T>
void MakePadding(const std::string& tf_padding_type,
                 const ngraph::Shape& ng_image_shape,
                 const ngraph::Shape& ng_kernel_shape,
                 const ngraph::Strides& ng_strides, T& ng_padding_below,
                 T& ng_padding_above) {
  if (tf_padding_type == "SAME") {
    for (size_t i = 0; i < 2; i++) {
      size_t image_size = ng_image_shape[i];
      size_t filter_shape = ng_kernel_shape[i];
      size_t filter_stride = ng_strides[i];

      int64 padding_needed;
      if (image_size % filter_stride == 0) {
        padding_needed = filter_shape - filter_stride;
      } else {
        padding_needed = filter_shape - (image_size % filter_stride);
      }
      if (padding_needed < 0) {
        padding_needed = 0;
      }

      size_t padding_lhs = padding_needed / 2;
      size_t padding_rhs = padding_needed - padding_lhs;
      ng_padding_below[i] = padding_lhs;
      ng_padding_above[i] = padding_rhs;
    }
  }

  NGRAPH_VLOG(3) << "ng_padding_below: " << ngraph::join(ng_padding_below);
  NGRAPH_VLOG(3) << "ng_padding_above: " << ngraph::join(ng_padding_above);
}

Status ValidateInputCount(const Node* op, size_t count) {
  if (op->num_inputs() != count) {
    return errors::InvalidArgument("\"", op->name(), "\" requires ", count,
                                   " input(s), got ", op->num_inputs(),
                                   " instead");
  }
  return Status::OK();
}

Status ValidateInputCountMin(const Node* op, size_t count) {
  if (op->num_inputs() < count) {
    return errors::InvalidArgument("\"", op->name(), "\" requires at least ",
                                   count, " input(s), got ", op->num_inputs(),
                                   " instead");
  }
  return Status::OK();
}

Status TranslateAddNOp(const Node* op, vector<shared_ptr<ng::Node>>& ng_arg_vec,
                       const std::vector<const Tensor*>& static_input_map,
                       vector<shared_ptr<ng::Node>>& subgraph_out_nodes) {
  subgraph_out_nodes[0] =
      std::accumulate(std::next(ng_arg_vec.begin()), ng_arg_vec.end(),
                      ng_arg_vec.at(0));  // accumulation: start with first
                                          // element. default op is addition
  return Status::OK();
}

// ng_arg_vec is not const. For example, BatchToNGraph changes it
Status TranslateAvgPoolOp(const Node* op,
                          vector<shared_ptr<ng::Node>>& ng_arg_vec,
                          const std::vector<const Tensor*>& static_input_map,
                          vector<shared_ptr<ng::Node>>& subgraph_out_nodes) {
  std::vector<int32> tf_strides;
  std::vector<int32> tf_ksize;
  std::string tf_padding_type;
  std::string tf_data_format;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "strides", &tf_strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "ksize", &tf_ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "padding", &tf_padding_type));
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "data_format", &tf_data_format));

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return errors::InvalidArgument(
        "AvgPool data format is neither NHWC nor NCHW");
  }

  bool is_nhwc = (tf_data_format == "NHWC");

  NGRAPH_VLOG(3) << ng::join(tf_strides);
  NGRAPH_VLOG(3) << ng::join(tf_ksize);
  NGRAPH_VLOG(3) << tf_padding_type;
  NGRAPH_VLOG(3) << tf_data_format;

  ng::Strides ng_strides(2);
  ng::Shape ng_image_shape(2);
  ng::Shape ng_kernel_shape(2);

  BatchedOpParamToNGraph(is_nhwc, tf_strides, ng_strides);
  BatchedOpParamToNGraph(is_nhwc, ng_arg_vec[0]->get_shape(), ng_image_shape);
  BatchedOpParamToNGraph(is_nhwc, tf_ksize, ng_kernel_shape);
  BatchToNGraph(is_nhwc, ng_arg_vec[0]);
  NGRAPH_VLOG(3) << "ng_strides: " << ng::join(ng_strides);
  NGRAPH_VLOG(3) << "ng_image_shape: " << ng::join(ng_image_shape);
  NGRAPH_VLOG(3) << "ng_kernel_shape: " << ng::join(ng_kernel_shape);

  // TODO: change this once nGraph supports negative padding
  // (CoordinateDiff) for AvgPool
  // ng::CoordinateDiff ng_padding_below{0,0};
  // ng::CoordinateDiff ng_padding_above{0,0};
  ng::Shape ng_padding_below{0, 0};
  ng::Shape ng_padding_above{0, 0};

  MakePadding(tf_padding_type, ng_image_shape, ng_kernel_shape, ng_strides,
              ng_padding_below, ng_padding_above);

  std::shared_ptr<ng::Node> ng_avgpool =
      make_shared<ng::op::AvgPool>(ng_arg_vec[0], ng_kernel_shape, ng_strides,
                                   ng_padding_below, ng_padding_above, false);

  BatchToTensorflow(is_nhwc, ng_avgpool);
  NGRAPH_VLOG(3) << "avgpool outshape: {" << ng::join(ng_avgpool->get_shape())
                 << "}";

  subgraph_out_nodes[0] = ng_avgpool;
  return Status::OK();
}

template <typename T, typename VecT = T>
Status MakeConstOp(const Node* op, ng::element::Type et,
                   std::shared_ptr<ng::Node>* ng_node) {
  vector<VecT> const_values;
  TensorShapeProto shape_proto;

  TF_RETURN_IF_ERROR(
      ValuesFromConstNode<T, VecT>(op->def(), &shape_proto, &const_values));

  TensorShape const_shape(shape_proto);
  ng::Shape ng_shape;
  TF_RETURN_IF_ERROR(TFTensorShapeToNGraphShape(const_shape, &ng_shape));

  *ng_node = make_shared<ng::op::Constant>(et, ng_shape, const_values);

  return Status::OK();
}

static Status TranslateConstOp(
    const Node* op, vector<shared_ptr<ng::Node>>& ng_arg_vec,
    const std::vector<const Tensor*>& static_input_map,
    vector<shared_ptr<ng::Node>>& subgraph_out_nodes) {
  const static std::map<
      const DataType,
      const std::pair<const std::function<Status(const Node*, ng::element::Type,
                                                 std::shared_ptr<ng::Node>*)>,
                      const ngraph::element::Type>>
      TF_NGRAPH_CONST_MAP = {
          {DataType::DT_FLOAT, make_pair(MakeConstOp<float>, ng::element::f32)},
          {DataType::DT_DOUBLE,
           make_pair(MakeConstOp<double>, ng::element::f64)},
          {DataType::DT_INT8, make_pair(MakeConstOp<int8>, ng::element::i8)},
          {DataType::DT_INT16, make_pair(MakeConstOp<int16>, ng::element::i16)},
          {DataType::DT_INT32, make_pair(MakeConstOp<int32>, ng::element::i32)},
          {DataType::DT_INT64, make_pair(MakeConstOp<int64>, ng::element::i64)},
          {DataType::DT_UINT8, make_pair(MakeConstOp<uint8>, ng::element::u8)},
          {DataType::DT_UINT16,
           make_pair(MakeConstOp<uint16>, ng::element::u16)},
          {DataType::DT_BOOL,
           make_pair(MakeConstOp<bool, char>, ng::element::boolean)}};

  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(op->attrs(), "dtype", &dtype));
  // For some reason the following do not work (no specialization of
  // tensorflow::checkpoint::SavedTypeTraits...)
  // case DataType::DT_UINT32:
  //   TF_RETURN_IF_ERROR(MakeConstOp<uint32>(op, ng::element::u32,
  //   &ng_node));
  //   break;
  // case DataType::DT_UINT64:
  //   TF_RETURN_IF_ERROR(MakeConstOp<uint64>(op, ng::element::u64,
  //   &ng_node));
  //   break;
  try {
    const auto& func_param = TF_NGRAPH_CONST_MAP.at(dtype);
    TF_RETURN_IF_ERROR(
        func_param.first(op, func_param.second, &subgraph_out_nodes[0]));
  } catch (const std::out_of_range&) {
    return errors::Unimplemented("Unsupported TensorFlow data type: ",
                                 DataType_Name(dtype));
  }
  return Status::OK();
}

Status TranslateFloorDivOp(const Node* op,
                           vector<shared_ptr<ng::Node>>& ng_arg_vec,
                           const std::vector<const Tensor*>& static_input_map,
                           vector<shared_ptr<ng::Node>>& subgraph_out_nodes) {
  subgraph_out_nodes[0] =
      std::make_shared<ng::op::Floor>(ng_arg_vec[0] / ng_arg_vec[1]);
  return Status::OK();
}

Status TranslateFloorModOp(const Node* op,
                           vector<shared_ptr<ng::Node>>& ng_arg_vec,
                           const std::vector<const Tensor*>& static_input_map,
                           vector<shared_ptr<ng::Node>>& subgraph_out_nodes) {
  auto floordiv =
      std::make_shared<ng::op::Floor>(ng_arg_vec[0] / ng_arg_vec[1]);
  subgraph_out_nodes[0] = ng_arg_vec[0] - (floordiv * ng_arg_vec[1]);

  // Possible reuse of floordiv:
  // TranslateFloorDivOp1(op, ng_arg_vec, static_input_map, subgraph_out_nodes);
  // subgraph_out_nodes[0] = ng_arg_vec[0] - (subgraph_out_nodes[0] *
  // ng_arg_vec[1]);
  return Status::OK();
}
}  // namespace ngraph_bridge
}  // namespace tensorflow