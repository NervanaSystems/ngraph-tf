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

#include <iostream>
#include <memory>

#include <cuda_runtime.h>

namespace tensorflow {
namespace ngraph_bridge {

template <typename T>
class TensorStream
{
  virtual std::ostream& stream(std::ostream&) const = 0;
  // template <typename V>
  // virtual void vector(std::vector<V>&) const = 0;
 public:
  TensorStream(const Tensor& tensor)
    : m_num_elements(tensor.NumElements())
    , m_tensor_ptr(reinterpret_cast<const T*>(tensor.tensor_data().data())) {;}
  virtual ~TensorStream() = default;
  friend std::ostream& operator<<(std::ostream& os, const TensorStream& t) {
    return t.stream(os);
  }

 protected:
  int64 m_num_elements;
  const T* m_tensor_ptr;
};

template <typename T>
class HostTensorStream : public TensorStream<T> {
 public:
  HostTensorStream(const Tensor& tensor)
    : TensorStream<T>(tensor) {;}
  virtual ~HostTensorStream() = default;
  template <typename V>
  void vector(std::vector<V>& vec) const {
    vec.resize(this->m_num_elements);
    for (int64 i = 0; i < this->m_num_elements; i++) {
      vec[i] = V(this->m_tensor_ptr[i]);
    }
  }
 private:
  virtual std::ostream& stream(std::ostream& ostream) const {
    for (int64 i = 0; i < this->m_num_elements; i++) {
      ostream << this->m_tensor_ptr[i] << ",";
    }
    return ostream;
  }
};

template <typename T>
class CUDADeviceTensorStream : public TensorStream<T> {
 public:
  CUDADeviceTensorStream(const Tensor& tensor)
    : TensorStream<T>(tensor) {;}
  virtual ~CUDADeviceTensorStream() = default;
  template <typename V>
  void vector(std::vector<V>& vec) const {
    size_t size = sizeof(T)*this->m_num_elements;
    T* buffer = static_cast<T*>(std::malloc(size));
    cudaMemcpy(buffer, this->m_tensor_ptr, size, cudaMemcpyDeviceToHost);
    vec.resize(this->m_num_elements);
    for (int64 i = 0; i < this->m_num_elements; i++) {
      vec[i] = V(buffer[i]);
    }
    free(buffer);
  }
 private:
  virtual std::ostream& stream(std::ostream& ostream) const {
    size_t size = sizeof(T)*this->m_num_elements;
    T* buffer = static_cast<T*>(std::malloc(size));
    cudaMemcpy(buffer, this->m_tensor_ptr, size, cudaMemcpyDeviceToHost);
    for (int64 i = 0; i < this->m_num_elements; i++) {
      ostream << buffer[i] << ",";
    }
    free(buffer);
    return ostream;
  }
};

static bool is_device_pointer(const void *ptr);

template <typename T>
std::unique_ptr<TensorStream<T>> TensorToStream(const Tensor& tensor) {
  if (is_device_pointer(tensor.tensor_data().data()))
  {
    return std::unique_ptr<CUDADeviceTensorStream<T>>(new CUDADeviceTensorStream<T>(tensor));
  }
  else
  {
    return std::unique_ptr<HostTensorStream<T>>(new HostTensorStream<T>(tensor));
  }
}

template <typename T, typename V>
void TensorToVector(const Tensor& tensor, std::vector<V>* vector) {
  if (is_device_pointer(tensor.tensor_data().data()))
  {
    CUDADeviceTensorStream<T>(tensor).vector(*vector);
  }
  else
  {
    HostTensorStream<T>(tensor).vector(*vector);
  }
}

static bool is_device_pointer(const void *ptr) {
  bool is_device_ptr = false;
  cudaPointerAttributes attributes;
  auto err = cudaPointerGetAttributes(&attributes, ptr);
  if(err != cudaSuccess)
  {
    err = cudaGetLastError();
    return is_device_ptr;
  }
  if(attributes.devicePointer != nullptr)
  {
    is_device_ptr = true;
  }
  return is_device_ptr;
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
