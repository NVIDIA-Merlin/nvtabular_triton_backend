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
#include "triton_utils.hpp"

namespace py = pybind11;

namespace triton {
namespace backend {
namespace nvtabular {
    
class NVT_LOCAL NVTabular {
 public:
  void Deserialize(const std::string &path_workflow,
                   const std::map<std::string, std::string> &dtypes) {
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
      
    store_column_types();
  }

  TRITONSERVER_Error* Transform(const std::vector<std::string>& input_names,
                 const std::vector<const void*>& input_buffers,
                 const std::vector<const int64_t*>& input_shapes,
                 const std::vector<TRITONSERVER_DataType>& input_dtypes,
                 const std::unordered_map<std::string, size_t>& max_str_sizes,
                 const std::vector<std::string>& output_names,
                 const std::vector<TRITONSERVER_DataType>& output_dtypes,
                 TRITONBACKEND_Response* response) {
    py::list all_inputs;
    py::list all_inputs_names;
    for (uint32_t i = 0; i < input_names.size(); ++i) {
      py::dict ai_in;
      std::tuple<int64_t> shape_in(static_cast<int64_t>(input_shapes[i][0]));
      ai_in["shape"] = shape_in;
      std::tuple<int64_t, bool> data_in((int64_t) * (&input_buffers[i]), false);
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
      if (column_types[output_names[i]]) {
        int64_t output_length = input_shapes[0][0];
        int64_t output_width = 1;
        int64_t output_byte_size = output_length * output_width *
            Utils::GetTritonTypeByteSize(output_dtypes[i]);

        std::vector<int64_t> batch_shape;
        batch_shape.push_back(output_length);
        batch_shape.push_back(output_width);

        TRITONBACKEND_Output* output_tri;
        TRITONBACKEND_ResponseOutput(
          response, &output_tri, output_names[i].c_str(), output_dtypes[i],
          batch_shape.data(), batch_shape.size());
          
        if (response == nullptr) {
          std::string error = (std::string("request ") + std::to_string(0) +
            ": failed to create response output, error response sent").c_str();
          LOG_MESSAGE(TRITONSERVER_LOG_ERROR, error.c_str());
          return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
            error.c_str());
        }

        TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
        int64_t output_memory_type_id = 0;
          
        void* output_buffer;
        TRITONBACKEND_OutputBuffer(
          output_tri, &output_buffer, output_byte_size, &output_memory_type,
          &output_memory_type_id);

        if ((response == nullptr) || (output_memory_type == TRITONSERVER_MEMORY_GPU)) {
          LOG(TRITONSERVER_LOG_ERROR) << "request " << 0
             <<  ": failed to create output buffer in CPU memory, error response sent";
          return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
            "failed to create output buffer in CPU memory");
        }
      }
    }

    py::tuple trans_data_info =
        nt.attr("transform")(all_inputs_names, all_inputs, all_output_names);
    py::dict output = trans_data_info[0];
    py::list lengths = trans_data_info[1];

    for (uint32_t i = 0; i < output_names.size(); ++i) {
      const char* output_name = output_names[i].c_str();
      int64_t output_length = lengths[i].cast<int64_t>();
      int64_t output_width = 1;
      int64_t output_byte_size = output_length * output_width *
          Utils::GetTritonTypeByteSize(output_dtypes[i]);

      std::vector<int64_t> batch_shape;
      batch_shape.push_back(output_length);
      batch_shape.push_back(output_width);

      TRITONBACKEND_Output* output_tri;
      TRITONBACKEND_ResponseOutput(
        response, &output_tri, output_name, output_dtypes[i],
        batch_shape.data(), batch_shape.size());

      if (response == nullptr) {
        std::string error = (std::string("request ") + std::to_string(0) +
                 ": failed to create response output, error response sent").c_str();
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, error.c_str());
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
          error.c_str());
      }

      TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t output_memory_type_id = 0;

      void* output_buffer;
      TRITONBACKEND_OutputBuffer(
        output_tri, &output_buffer, output_byte_size, &output_memory_type,
        &output_memory_type_id);

      if ((response == nullptr) || (output_memory_type == TRITONSERVER_MEMORY_GPU)) {
        LOG(TRITONSERVER_LOG_ERROR) << "request " << 0
           <<  ": failed to create output buffer in CPU memory, error response sent";
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
          "failed to create output buffer in CPU memory");
      }

      if (output_dtypes[i] == TRITONSERVER_TYPE_BOOL) {
        py::array_t<bool> arr =
            (py::array_t<bool>)output[output_names[i].c_str()];
        memcpy(output_buffer, arr.data(), output_byte_size);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_UINT8) {
        py::array_t<uint8_t> arr =
            (py::array_t<uint8_t>)output[output_names[i].c_str()];
        memcpy(output_buffer, arr.data(), output_byte_size);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_UINT16) {
        py::array_t<uint16_t> arr =
            (py::array_t<uint16_t>)output[output_names[i].c_str()];
        memcpy(output_buffer, arr.data(), output_byte_size);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_UINT32) {
        py::array_t<uint32_t> arr =
            (py::array_t<uint32_t>)output[output_names[i].c_str()];
        memcpy(output_buffer, arr.data(), output_byte_size);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_UINT64) {
        py::array_t<uint64_t> arr =
            (py::array_t<uint64_t>)output[output_names[i].c_str()];
        memcpy(output_buffer, arr.data(), output_byte_size);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_INT8) {
        py::array_t<int8_t> arr =
            (py::array_t<int8_t>)output[output_names[i].c_str()];
        memcpy(output_buffer, arr.data(), output_byte_size);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_INT16) {
        py::array_t<int16_t> arr =
            (py::array_t<int16_t>)output[output_names[i].c_str()];
        memcpy(output_buffer, arr.data(), output_byte_size);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_INT32) {
        py::array_t<int32_t> arr =
            (py::array_t<int32_t>)output[output_names[i].c_str()];
        memcpy(output_buffer, arr.data(), output_byte_size);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_INT64) {
        py::array_t<uint64_t> arr =
            (py::array_t<uint64_t>)output[output_names[i].c_str()];
        memcpy(output_buffer, arr.data(), output_byte_size);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_FP16) {
        throw std::invalid_argument("Unhandled dtype: fp16");
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_FP32) {
        py::array_t<float> arr =
            (py::array_t<float>)output[output_names[i].c_str()];
        memcpy(output_buffer, arr.data(), output_byte_size);
      } else if (output_dtypes[i] == TRITONSERVER_TYPE_FP64) {
        py::array_t<double> arr =
            (py::array_t<double>)output[output_names[i].c_str()];
        memcpy(output_buffer, arr.data(), output_byte_size);
      } else {
        throw std::invalid_argument("Unhandled dtype");
      }
    }
    return nullptr;
  }

  std::unordered_map<std::string, bool> GetColumnTypes() {
    return column_types;
  }

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
    
  void store_column_types() {
    py::dict col_types = nt.attr("get_column_types")();      
    for (auto item : col_types) {
      if (std::string(py::str(item.second)).compare("ColumnType.SINGLEHOT") == 0) {
        column_types[std::string(py::str(item.first))] = true;
      } else {
        column_types[std::string(py::str(item.first))] = false;
      }
    }
  }

  std::unordered_map<std::string, bool> column_types;
  std::map<std::string, std::string> dtypes;
  py::object nt;
};

} // namespace nvtabular
} // namespace backend
} // namespace triton
#endif // NVTABULAR_HPP_
