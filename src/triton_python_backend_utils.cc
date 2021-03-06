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

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "triton_utils.hpp"
#include "triton_python_backend_utils.hpp"


namespace triton {
namespace backend {
namespace nvtabular {
// Export our c++ classes to python as a 'triton_python_backend_utils' builtin module
PYBIND11_EMBEDDED_MODULE(triton_python_backend_utils, m) {
  py::class_<InferenceResponse>(m, "InferenceResponse")
    .def(py::init<py::list, py::object>(), py::arg("tensors"), py::arg("error") = py::none());

  py::class_<InferenceRequest>(m, "InferenceRequest")
    .def(py::init<>())
    .def("request_id", &InferenceRequest::get_request_id)
    .def("correlation_id", &InferenceRequest::get_correlation_id)
    .def("inputs", &InferenceRequest::get_inputs, py::return_value_policy::reference);

  py::class_<Tensor>(m, "Tensor")
    .def(py::init<std::string, py::object>())
    .def_readonly("as_numpy", &Tensor::numpy_array)
    .def_readonly("name", &Tensor::name);

  // the triton_python_backend class uses the single 'Tensor' class for both inputs and outputs
  // but our implementation is much easier if we separate out inputs from outputs (maps
  // closer to the triton c++ backend api). Since Tensor objects are  only constructed from python
  // for outputs, and the inputs are passed in, this won't cause any issues (unless the triton
  // python model is checking the class of the inputs, which would be kinda weird).
  py::class_<Input>(m, "Input")
    .def_readonly("name", &Input::name)
    .def("as_numpy", input_as_numpy);

  // Define the 'TRITON_SERVER_TO_NUMPY' global
  py::object np = py::module_::import("numpy");
  py::dict TRITON_STRING_TO_NUMPY;
  TRITON_STRING_TO_NUMPY["TYPE_BOOL"] = np.attr("bool");
  TRITON_STRING_TO_NUMPY["TYPE_UINT8"] = np.attr("uint8");
  TRITON_STRING_TO_NUMPY["TYPE_UINT16"] = np.attr("uint16");
  TRITON_STRING_TO_NUMPY["TYPE_UINT32"] = np.attr("uint32");
  TRITON_STRING_TO_NUMPY["TYPE_UINT64"] = np.attr("uint64");
  TRITON_STRING_TO_NUMPY["TYPE_INT8"] = np.attr("int8");
  TRITON_STRING_TO_NUMPY["TYPE_INT16"] = np.attr("int16");
  TRITON_STRING_TO_NUMPY["TYPE_INT32"] = np.attr("int32");
  TRITON_STRING_TO_NUMPY["TYPE_INT64"] = np.attr("int64");
  TRITON_STRING_TO_NUMPY["TYPE_FP16"] = np.attr("float16");
  TRITON_STRING_TO_NUMPY["TYPE_FP32"] = np.attr("float32");
  TRITON_STRING_TO_NUMPY["TYPE_FP64"] = np.attr("float64");
  TRITON_STRING_TO_NUMPY["TYPE_STRING"] = np.attr("object_");
  m.attr("TRITON_STRING_TO_NUMPY") = TRITON_STRING_TO_NUMPY;

  m.def("triton_string_to_numpy", [TRITON_STRING_TO_NUMPY](py::object triton_string) {
    return TRITON_STRING_TO_NUMPY[triton_string];
  });
  m.def("get_input_tensor_by_name", get_input_tensor_by_name, py::return_value_policy::reference);
  m.def("get_output_config_by_name", get_output_config_by_name);

  py::bind_vector<std::vector<std::string>>(m, "StringVector");
  py::bind_vector<std::vector<Input>>(m, "InputVector");
  py::bind_vector<std::vector<InferenceRequest>>(m, "InferenceRequestVector");
}

py::array input_as_numpy(const Input & tensor) {
  if (tensor.memory_type == TRITONSERVER_MEMORY_GPU) {
    throw std::invalid_argument("Can't convert GPU tensor to numpy");
  }

  if (tensor.dtype == TRITONSERVER_TYPE_BYTES) {
    std::vector<std::string> values;
    const char * bytes = reinterpret_cast<const char *>(tensor.buffer);
    for (uint64_t i = 0; i < tensor.buffer_size;) {
      int size = *reinterpret_cast<const int *>(bytes + i);
      std::string value(bytes + i + sizeof(int), size);
      values.push_back(value);
      i += size + sizeof(size);
    }
    return py::array(py::cast(values));
  }

  // Return a numpy array thats a view of this tensor (without copying)
  // https://github.com/pybind/pybind11/issues/2271#issuecomment-768175261
  pybind11::dtype dtype(triton_dtype_to_numpy_typestr(tensor.dtype));
  pybind11::array::ShapeContainer shape(tensor.shape, tensor.shape + tensor.dims);
  return py::array(dtype, shape, tensor.buffer);
}

const char * triton_dtype_to_numpy_typestr(TRITONSERVER_DataType dtype) {
  switch (dtype) {
    case TRITONSERVER_TYPE_INT8:
      return "<i1";
    case TRITONSERVER_TYPE_UINT8:
      return "<u1";
    case TRITONSERVER_TYPE_INT16:
      return "<i2";
    case TRITONSERVER_TYPE_UINT16:
      return "<u2";
    case TRITONSERVER_TYPE_INT32:
      return "<i4";
    case TRITONSERVER_TYPE_UINT32:
      return "<u4";
    case TRITONSERVER_TYPE_INT64:
      return "<i8";
    case TRITONSERVER_TYPE_UINT64:
      return "<u8";
    case TRITONSERVER_TYPE_FP16:
      return "<f2";
    case TRITONSERVER_TYPE_FP32:
      return "<f4";
    case TRITONSERVER_TYPE_FP64:
      return "<f8";
    case TRITONSERVER_TYPE_BOOL:
      return "|b1";
    default:
      throw std::invalid_argument("unhandled dtype");
  }
}

TRITONSERVER_DataType numpy_to_triton_dtype(char kind, int itemsize) {
  switch (kind) {
    case 'i':
      switch (itemsize) {
        case 1: return TRITONSERVER_TYPE_INT8;
        case 2: return TRITONSERVER_TYPE_INT16;
        case 4: return TRITONSERVER_TYPE_INT32;
        case 8: return TRITONSERVER_TYPE_INT64;
      }
      break;
    case 'u':
      switch (itemsize) {
        case 1: return TRITONSERVER_TYPE_UINT8;
        case 2: return TRITONSERVER_TYPE_UINT16;
        case 4: return TRITONSERVER_TYPE_UINT32;
        case 8: return TRITONSERVER_TYPE_UINT64;
      }
      break;
    case 'f':
      switch (itemsize) {
        case 2: return TRITONSERVER_TYPE_FP16;
        case 4: return TRITONSERVER_TYPE_FP32;
        case 8: return TRITONSERVER_TYPE_FP64;
      }
      break;
    case 'b':
      return TRITONSERVER_TYPE_BOOL;
      break;
  }

  // don't know how to handle this typestr
  std::stringstream err;
  err << "Unhandled numpy dtype: kind " << kind << " itemsize " << itemsize;
  throw std::invalid_argument(err.str());
}

const Input * get_input_tensor_by_name(const InferenceRequest & request, const std::string & name) {
  auto inputs = request.get_inputs();
  for (auto it = inputs->begin(); it != inputs->end(); ++it) {
    if (name == it->name) {
      return &*it;
    }
  }
  return NULL;
}

py::object get_output_config_by_name(py::dict model_config, py::str name) {
  if (model_config.contains("output")) {
    py::object outputs = model_config["output"];
    for (auto & output_properties : outputs) {
      py::str output_name = output_properties["name"];
      if (output_name.equal(name)) {
        return py::reinterpret_borrow<py::object>(output_properties);
      }
    }
  }
  return py::none();
}

}  // namespace nvtabular
}  // namespace backend
}  // namespace triton

PYBIND11_MAKE_OPAQUE(std::vector<triton::backend::nvtabular::Input>);
PYBIND11_MAKE_OPAQUE(std::vector<triton::backend::nvtabular::InferenceRequest>);
PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
