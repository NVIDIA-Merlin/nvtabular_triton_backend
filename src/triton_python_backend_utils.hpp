// Copyright (c) 2021, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TRITON_PYTHON_BACKEND_UTILS_HPP_
#define TRITON_PYTHON_BACKEND_UTILS_HPP_
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "triton_utils.hpp"

namespace py = pybind11;


namespace triton {
namespace backend {
namespace nvtabular {
// we're mimicing the API from the python backend:
// https://github.com/triton-inference-server/python_backend/blob/main/src/resources/triton_python_backend_utils.py
// by creating a pybind11 embedded module. This will let python code treat this backend exactly
// the same as the existing python backend - with the only difference being that this will run in
// the same process as the tritonserver

const char * triton_dtype_to_numpy_typestr(TRITONSERVER_DataType dtype);
TRITONSERVER_DataType numpy_to_triton_dtype(char kind, int itemsize);
const Input * get_input_tensor_by_name(const InferenceRequest & request, const std::string & name);
py::object get_output_config_by_name(py::dict model_config, py::str name);

// Holds onto numpy tensors being output by the python model file, and converts to triton output
// Mimics the 'Tensor' class in the python_backend
// https://github.com/triton-inference-server/python_backend/blob/fd0f6ba090ce/src/resources/triton_python_backend_utils.py#L235  NOLINT
class NVT_LOCAL Tensor {
 public:
  Tensor(const std::string & name, py::array numpy_array)
    : name(name), numpy_array(numpy_array) {
  }

  void copy_to_triton(TRITONBACKEND_Response * response) {
    // Create the triton output
    auto shape = numpy_array.shape();
    auto ndim = numpy_array.ndim();
    auto numpy_dtype = numpy_array.dtype();
    auto triton_dtype = numpy_to_triton_dtype(numpy_dtype.kind(), numpy_dtype.itemsize());
    auto byte_size = numpy_array.nbytes();
    TRITONBACKEND_Output* triton_output;
    check_triton(
      TRITONBACKEND_ResponseOutput(response, &triton_output, name.c_str(), triton_dtype, shape, ndim));

    // copy the numpy array over to the output
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    void * buffer;
    check_triton(TRITONBACKEND_OutputBuffer(triton_output, &buffer, byte_size, &memory_type, &memory_type_id));
    memcpy(buffer, numpy_array.data(), byte_size);
  }

  std::string name;
  py::array numpy_array;
  Tensor(Tensor &&) = default;
  Tensor& operator=(Tensor && other) = default;
  Tensor(const Tensor &) = delete;
  Tensor& operator=(const Tensor &) = delete;
};

// Container for all the Tensor objects in a response, and mimics the  'InferenceResponse' object in the python_backend
// https://github.com/triton-inference-server/python_backend/blob/fd0f6ba090ce5c9ed8938da08bceb1d407e5b33a/src/resources/triton_python_backend_utils.py#L186 NOLINT
struct NVT_LOCAL InferenceResponse {
  InferenceResponse(py::list tensors, py::object error) : tensors(tensors), error(error) { }

  void copy_to_triton(TRITONBACKEND_Response * response) {
    for (auto & output : tensors) {
      py::cast<Tensor &>(output).copy_to_triton(response);
    }
  }

  py::list tensors;
  py::object error;

  InferenceResponse(InferenceResponse &&) = default;
  InferenceResponse& operator=(InferenceResponse && other) = default;
  InferenceResponse(const InferenceResponse &) = delete;
  InferenceResponse& operator=(const InferenceResponse &) = delete;
};

py::array input_as_numpy(const Input & tensor);
}  // namespace nvtabular
}  // namespace backend
}  // namespace triton
#endif  // TRITON_PYTHON_BACKEND_UTILS_HPP_
