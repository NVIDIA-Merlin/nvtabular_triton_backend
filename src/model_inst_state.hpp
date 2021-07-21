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

#ifndef MODEL_INST_STATE_HPP_
#define MODEL_INST_STATE_HPP_

#include <pybind11/embed.h>

#include <map>
#include <string>
#include <vector>

#include "triton_utils.hpp"
#include "triton_python_backend_utils.hpp"


namespace triton {
namespace backend {
namespace nvtabular {

namespace py = pybind11;


class NVT_LOCAL ModelInstanceState {
 public:
  explicit ModelInstanceState(TRITONBACKEND_ModelInstance * instance)
    : instance_(instance) {
    const char *instance_name;
    check_triton(TRITONBACKEND_ModelInstanceName(instance, &instance_name));
    check_triton(TRITONBACKEND_ModelInstanceKind(instance, &kind_));
    check_triton(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id_));
    name_ = instance_name;

    TRITONBACKEND_Model* model;
    check_triton(TRITONBACKEND_ModelInstanceModel(instance, &model));
    check_triton(TRITONBACKEND_ModelState(model, reinterpret_cast<void**>(&model_state_)));

    // Create the TritonPythonModel and initialize it from the model state
    py::gil_scoped_acquire l;
    auto python_module = model_state_->PythonModule();
    auto path = model_state_->Path();
    auto version = model_state_->Version();

    py::object module;
    if (python_module.size() > 0) {
      // if we've been given a python module name in the model_config parameters use that
      LOG(TRITONSERVER_LOG_INFO) << "Loading TritonPythonModel from module '" << python_module << "'";
      module = py::module_::import(python_module.c_str());
    } else {
      // otherwise default to the 'model.py' file bundled with the triton model
      std::stringstream model_path;
      model_path << path << "/" << version;
      LOG(TRITONSERVER_LOG_INFO) << "Loading TritonPythonnModel from model.py in path '" << model_path.str() << "'";
      py::object sys = py::module_::import("sys");
      sys.attr("path").attr("insert")(0, model_path.str());
      module = py::module_::import("model");
    }

    // initialize the model
    python_model = module.attr("TritonPythonModel")();
    py::dict args;
    args["model_config"] = model_state_->ModelConfig();
    args["model_version"] = model_state_->Version();
    args["model_name"] = model_state_->Name();
    args["model_repository"] = model_state_->Path();
    args["model_instance_kind"] = TRITONSERVER_InstanceGroupKindString(kind_);
    args["model_instance_name"] = name_;
    args["model_instance_device_id"] = std::to_string(device_id_);
    python_model.attr("initialize")(args);
  }

  void transform_requests(TRITONBACKEND_Request ** triton_requests,
                          TRITONBACKEND_Response ** triton_responses,
                          uint32_t request_count) {
    uint64_t exec_start = timestamp_ns();

    std::vector<InferenceRequest> requests;
    for (uint32_t i = 0; i < request_count; ++i) {
      requests.push_back(InferenceRequest(triton_requests[i]));
    }

    std::vector<TRITONSERVER_Error *> errors(request_count, nullptr);

    uint64_t compute_start = timestamp_ns(), compute_end = 0;
    {
      // Transform the request using the python model. We need the GIL here, so this is scoped as tightly
      // as possible with the GIL to reduce contention.
      py::gil_scoped_acquire l;
      py::list responses = python_model.attr("execute")(py::cast(&requests, py::return_value_policy::reference));
      if (py::len(responses) != request_count) {
        throw std::invalid_argument("number of responses doesn't match number of requests");
      }
      compute_end = timestamp_ns();

      // copy the outputs out from python back the the TRITONBACKED_Response object
      for (uint32_t i = 0; i < request_count; ++i) {
        auto & response = py::cast<InferenceResponse&>(responses[i]);
        if (response.error.is_none()) {
          response.copy_to_triton(triton_responses[i]);
        } else {
          auto error_text =  py::cast<std::string>(response.error);
          errors[i] = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, error_text.c_str());
        }
      }
      // GIL is released here - no further python object access in this function is allowed
    }

    // send responses. doing this out of previous 'copy' loop to avoid holding GIL during this
    // operation.
    for (uint32_t i = 0; i < request_count; ++i) {
      auto err = TRITONBACKEND_ResponseSend(triton_responses[i], TRITONSERVER_RESPONSE_COMPLETE_FINAL, errors[i]);
      LOG_IF_ERROR(err, "failed sending response");
    }

    uint64_t exec_end = timestamp_ns();

    // log timing statistics for this request, and release it
    for (uint32_t i = 0; i < request_count; ++i) {
      auto request = triton_requests[i];
      bool success = errors[i] == nullptr;
      auto err = TRITONBACKEND_ModelInstanceReportStatistics(instance_, request, success,
                                                             exec_start, compute_start,
                                                             compute_end, exec_end);
      LOG_IF_ERROR(err, "failed to report request statistics");

      err = TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL);
      LOG_IF_ERROR(err, "failed releasing request");
    }

    // report batch statistics. this backend (like the triton python_backend) doesn't support batching
    // so the batch size is always 1
    auto err = TRITONBACKEND_ModelInstanceReportBatchStatistics(instance_, 1,
                                                                exec_start, compute_start,
                                                                compute_end, exec_end);
    LOG_IF_ERROR(err, "failed reporting batch request statistics");
  }

 private:
  ModelState *model_state_;
  TRITONBACKEND_ModelInstance * instance_;
  std::string name_;
  TRITONSERVER_InstanceGroupKind kind_;
  int32_t device_id_;

  py::object python_model;
};
}  // namespace nvtabular
}  // namespace backend
}  // namespace triton

#endif  // MODEL_INST_STATE_HPP_
