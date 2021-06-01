// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef NVTABULAR_HPP_
#define NVTABULAR_HPP_

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include <exception>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "triton/backend/backend_common.h"
#include "utils.hpp"

namespace py = pybind11;

namespace triton {
namespace backend {
namespace nvtabular {

class NVT_LOCAL NVTabular {
 private:
  void fill_array_interface(py::dict & ai, const size_t max_size) {  // NOLINT
    py::list list_desc;
    std::string u("<U");
    u.append(std::to_string(max_size));
    ai["typestr"] = u.c_str();
    std::tuple<std::string, std::string> desc("", u.c_str());
    list_desc.append(desc);
    ai["descr"] = list_desc;
    ai["version"] = 3;
  }

  void fill_array_interface(py::dict & ai, TRITONSERVER_DataType dtype) {  // NOLINT
    py::list list_desc;
    if (dtype == TRITONSERVER_TYPE_BOOL) {
      ai["typestr"] = "|b1";
      std::tuple<std::string, std::string> desc("", "|b1");
      list_desc.append(desc);
    } else if (dtype == TRITONSERVER_TYPE_INT8) {
      ai["typestr"] = "<i1";
      std::tuple<std::string, std::string> desc("", "<i1");
      list_desc.append(desc);
    } else if (dtype == TRITONSERVER_TYPE_INT16) {
      ai["typestr"] = "<i2";
      std::tuple<std::string, std::string> desc("", "<i2");
      list_desc.append(desc);
    } else if (dtype == TRITONSERVER_TYPE_INT32) {
      ai["typestr"] = "<i4";
      std::tuple<std::string, std::string> desc("", "<i4");
      list_desc.append(desc);
    } else if (dtype == TRITONSERVER_TYPE_INT64) {
      ai["typestr"] = "<i8";
      std::tuple<std::string, std::string> desc("", "<i8");
      list_desc.append(desc);
    } else if (dtype == TRITONSERVER_TYPE_UINT8) {
      ai["typestr"] = "<i1";
      std::tuple<std::string, std::string> desc("", "<u1");
      list_desc.append(desc);
    } else if (dtype == TRITONSERVER_TYPE_UINT16) {
      ai["typestr"] = "<i2";
      std::tuple<std::string, std::string> desc("", "<u2");
      list_desc.append(desc);
    } else if (dtype == TRITONSERVER_TYPE_UINT32) {
      ai["typestr"] = "<i4";
      std::tuple<std::string, std::string> desc("", "<u4");
      list_desc.append(desc);
    } else if (dtype == TRITONSERVER_TYPE_UINT64) {
      ai["typestr"] = "<i8";
      std::tuple<std::string, std::string> desc("", "<u8");
      list_desc.append(desc);
    } else if (dtype == TRITONSERVER_TYPE_FP16) {
      ai["typestr"] = "<f2";
      std::tuple<std::string, std::string> desc("", "<f2");
      list_desc.append(desc);
    } else if (dtype == TRITONSERVER_TYPE_FP32) {
      ai["typestr"] = "<f4";
      std::tuple<std::string, std::string> desc("", "<f4");
      list_desc.append(desc);
    } else if (dtype == TRITONSERVER_TYPE_FP64) {
      ai["typestr"] = "<f8";
      std::tuple<std::string, std::string> desc("", "<f8");
      list_desc.append(desc);
    }
    ai["descr"] = list_desc;
    ai["version"] = 3;
  }

  std::map<std::string, std::string> dtypes;

 public:
  void Deserialize(const std::string &path_workflow,
                   const std::map<std::string, std::string> & dtypes) {
    this->dtypes = dtypes;

    py::dict dtypes_py;
    for (auto it = dtypes.begin(); it != dtypes.end(); ++it) {
      dtypes_py[py::str(it->first)] = it->second;
    }

    py::object nvtabular =
        py::module_::import("nvtabular.inference.triton.backend_tf")
            .attr("TritonNVTabularModel");

    nt = nvtabular();
    nt.attr("initialize")(path_workflow.data(), dtypes_py);
  }

  void Transform(const std::vector<std::string> &input_names,
                 const void **input_buffers, const int64_t **input_shapes,
                 TRITONSERVER_DataType *input_dtypes,
                 const std::unordered_map<std::string, size_t> &max_str_sizes,
                 const std::vector<std::string> &output_names) {
    py::list all_inputs;
    py::list all_inputs_names;
    for (uint32_t i = 0; i < input_names.size(); ++i) {
      py::dict ai_in;
      std::tuple<int64_t> shape_in((int64_t)input_shapes[i][0]);
      ai_in["shape"] = shape_in;
      std::tuple<int64_t, bool> data_in((int64_t)*(&input_buffers[i]), false);
      ai_in["data"] = data_in;
      if (input_dtypes[i] == TRITONSERVER_TYPE_BYTES) {
        fill_array_interface(ai_in, max_str_sizes.at(input_names[i]));
      } else {
        fill_array_interface(ai_in, input_dtypes[i]);
      }
      all_inputs.append(ai_in);
      all_inputs_names.append(input_names[i]);
    }

    py::list all_output_names;
    for (uint32_t i = 0; i < output_names.size(); ++i) {
      all_output_names.append(output_names[i]);
    }

    output =
        nt.attr("transform")(all_inputs_names, all_inputs, all_output_names);
  }

  void CopyData(void **output_buffers, const uint64_t *output_byte_sizes,
                const std::vector<std::string> &output_names,
                const std::vector<TRITONSERVER_DataType> &output_dtypes) {
    for (uint32_t i = 0; i < output_names.size(); ++i) {
      if (output_dtypes[i] == TRITONSERVER_TYPE_BOOL) {
        py::array_t<bool> arr =
            (py::array_t<bool>)output[output_names[i].c_str()];
        memcpy(output_buffers[i], arr.data(), output_byte_sizes[i]);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_UINT8) {
        py::array_t<uint8_t> arr =
            (py::array_t<uint8_t>)output[output_names[i].c_str()];
        memcpy(output_buffers[i], arr.data(), output_byte_sizes[i]);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_UINT16) {
        py::array_t<uint16_t> arr =
            (py::array_t<uint16_t>)output[output_names[i].c_str()];
        memcpy(output_buffers[i], arr.data(), output_byte_sizes[i]);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_UINT32) {
        py::array_t<uint32_t> arr =
            (py::array_t<uint32_t>)output[output_names[i].c_str()];
        memcpy(output_buffers[i], arr.data(), output_byte_sizes[i]);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_UINT64) {
        py::array_t<uint64_t> arr =
            (py::array_t<uint64_t>)output[output_names[i].c_str()];
        memcpy(output_buffers[i], arr.data(), output_byte_sizes[i]);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_INT8) {
        py::array_t<int8_t> arr =
            (py::array_t<int8_t>)output[output_names[i].c_str()];
        memcpy(output_buffers[i], arr.data(), output_byte_sizes[i]);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_INT16) {
        py::array_t<int16_t> arr =
            (py::array_t<int16_t>)output[output_names[i].c_str()];
        memcpy(output_buffers[i], arr.data(), output_byte_sizes[i]);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_INT32) {
        py::array_t<int32_t> arr =
            (py::array_t<int32_t>)output[output_names[i].c_str()];
        memcpy(output_buffers[i], arr.data(), output_byte_sizes[i]);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_INT64) {
        py::array_t<uint64_t> arr =
            (py::array_t<uint64_t>)output[output_names[i].c_str()];
        memcpy(output_buffers[i], arr.data(), output_byte_sizes[i]);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_FP16) {
        throw std::invalid_argument("Unhandled dtype: fp16");
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_FP32) {
        py::array_t<float> arr =
            (py::array_t<float>)output[output_names[i].c_str()];
        memcpy(output_buffers[i], arr.data(), output_byte_sizes[i]);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_FP64) {
        py::array_t<double> arr =
            (py::array_t<double>)output[output_names[i].c_str()];
        memcpy(output_buffers[i], arr.data(), output_byte_sizes[i]);
      } else {
      }
    }
  }

  py::list GetOutputSizes() {
    return nt.attr("get_lengths")();
  }

 private:
  py::object nt;
  py::dict output;
};

}  // namespace nvtabular
}  // namespace backend
}  // namespace triton
#endif  // NVTABULAR_HPP_
