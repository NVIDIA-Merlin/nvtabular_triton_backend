// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <dlfcn.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <rapidjson/document.h>

#include <memory>
#include <thread>
#include <unordered_map>
#include <chrono>
#include <iostream>
#include <fstream>

#include "triton/backend/backend_common.h"
#include "nvtabular.hpp"
#include "model_state.hpp"
#include "model_inst_state.hpp"
#include "utils.hpp"
#include "triton_utils.hpp"

namespace triton { namespace backend { namespace nvtabular {

//
// Simple backend that demonstrates the TRITONBACKEND API for a
// blocking backend. A blocking backend completes execution of the
// inference before returning from TRITONBACKED_ModelInstanceExecute.
//
// This backend supports any model that has exactly 1 input and
// exactly 1 output. The input and output can have any name, datatype
// and shape but the shape and datatype of the input and output must
// match. The backend simply responds with the output tensor equal to
// the input tensor.
//

extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend.
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG(TRITONSERVER_LOG_INFO) << "TRITONBACKEND_Initialize: " << name;

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG(TRITONSERVER_LOG_INFO) << "Triton TRITONBACKEND API version: " << api_version_major << "."
    << api_version_minor;

  LOG(TRITONSERVER_LOG_INFO) << "'" << name << "' TRITONBACKEND API version: " <<
    TRITONBACKEND_API_VERSION_MAJOR << "." << TRITONBACKEND_API_VERSION_MINOR;

  /*
  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }
  */

  // Force opening of libpython - so that it's available globally for c-extension modules
  std::stringstream python_lib;
  python_lib << "libpython" << PY_MAJOR_VERSION << "." << PY_MINOR_VERSION << ".so";
  void *handle = dlopen(python_lib.str().c_str(), RTLD_LAZY | RTLD_GLOBAL);
  if (!handle) {
    LOG(TRITONSERVER_LOG_INFO) << "Failed to dlopen '" << python_lib.str() << "': " << dlerror();
    return TRITONSERVER_ErrorNew(
       TRITONSERVER_ERROR_INTERNAL,
       dlerror());
  } else {
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Loaded libpython successfully");
  }

  py::initialize_interpreter();
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Python interpreter is initialized");

  // we have to manually release the GIL here otherwise we'll deadlock in future threads
  PyEval_SaveThread();

  return nullptr;  // success
}

// Implementing TRITONBACKEND_Finalize is optional unless state is set
// using TRITONBACKEND_BackendSetState. The backend must free this
// state and perform any other global cleanup.
TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend) {
  try {
    if (Py_IsInitialized()) {
      py::gil_scoped_acquire l;
      py::finalize_interpreter();
    }
    return nullptr;
  } catch (const std::exception & e) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, e.what());
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.what());
  }
}

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG(TRITONSERVER_LOG_INFO) << "TRITONBACKEND_ModelInitialize: " << name << " (version " <<
    version << ")";

  // Can get location of the model artifacts. Normally we would need
  // to check the artifact type to make sure it was something we can
  // handle... but we are just going to log the location so we don't
  // need the check. We would use the location if we wanted to load
  // something from the model's repo.
  TRITONBACKEND_ArtifactType artifact_type;
  const char* clocation;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelRepository(model, &artifact_type, &clocation));

  LOG(TRITONSERVER_LOG_INFO) << "Repository location "  << clocation;

  // The model can access the backend as well... here we can access
  // the backend global state.
  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  RETURN_IF_ERROR(model_state->ReadInputOutputNames());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
  ModelState * model_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, reinterpret_cast<void**>(&model_state)));
  LOG(TRITONSERVER_LOG_INFO) << "TRITONBACKEND_ModelFinalize: delete model state";
  delete model_state;
  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance) {
  const char* cname;
  int32_t device_id;
  TRITONSERVER_InstanceGroupKind kind;

  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  std::string name(cname);
  LOG(TRITONSERVER_LOG_INFO) << "TRITONBACKEND_ModelInstanceInitialize: " << name << " ("
       << TRITONSERVER_InstanceGroupKindString(kind) << " device " << device_id << ")";

  // The instance can access the corresponding model as well... here
  // we get the model and from that get the model's state.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  ModelState* model_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, reinterpret_cast<void**>(&model_state)));

  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance) {
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG(TRITONSERVER_LOG_INFO) << "TRITONBACKEND_ModelInstanceFinalize: delete instance state";

  delete instance_state;

  return nullptr;  // success
}

void transform_request(ModelInstanceState * instance_state, TRITONBACKEND_Request * request,
                       TRITONBACKEND_Response * response) {
  const char* request_id = "";
  check_triton(TRITONBACKEND_RequestId(request, &request_id));

  uint64_t correlation_id = 0;
  check_triton(TRITONBACKEND_RequestCorrelationId(request, &correlation_id));

  uint32_t input_count = 0, output_count = 0;
  check_triton(TRITONBACKEND_RequestInputCount(request, &input_count));
  check_triton(TRITONBACKEND_RequestOutputCount(request, &output_count));

  LOG(TRITONSERVER_LOG_INFO) << "request_id = '"  << request_id
      << "', correlation_id = " << correlation_id << ", input_count " << input_count
      << ", output_count = " << output_count;

  // TODO(benfred) convert to input structure
  std::vector<TRITONBACKEND_Input*> inputs(input_count);
  std::vector<TRITONSERVER_DataType> input_dtypes(input_count);
  std::vector<const int64_t*> input_shapes(input_count);
  std::vector<uint32_t> input_dims_counts(input_count);
  std::vector<uint64_t> input_byte_sizes(input_count);
  std::vector<uint32_t> input_buffer_counts(input_count);
  std::vector<std::unique_ptr<std::vector<wchar_t>>> numpy_input_buffers;
  std::unordered_map<std::string, size_t> max_str_sizes;
  std::vector<const void*> input_buffers(input_count);
  std::vector<uint64_t> buffer_byte_sizes(input_count);

  ModelState* model_state = instance_state->StateForModel();
  const std::vector<std::string> & input_names = model_state->InputNames();

  for (uint32_t i = 0; i < input_count; i++) {
    const char* input_name = input_names[i].c_str();
    check_triton(TRITONBACKEND_RequestInput(request, input_name, &inputs[i]));
    check_triton(TRITONBACKEND_InputProperties(
      inputs[i], &input_name, &input_dtypes[i], &input_shapes[i],
      &input_dims_counts[i], &input_byte_sizes[i], &input_buffer_counts[i]));

    if (input_buffer_counts[i] != 1) {
      std::stringstream err;
      err << "input_buffer_count " << input_buffer_counts[i] << " not supported for input " << input_names[i];
      throw std::invalid_argument(err.str());
    }

    input_buffers[i] = nullptr;

    buffer_byte_sizes[i] = 0;
    TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t input_memory_type_id = 0;

    const void* input_buffer;

    check_triton(TRITONBACKEND_InputBuffer(
        inputs[i], 0, &input_buffer, &buffer_byte_sizes[i], &input_memory_type,
        &input_memory_type_id));

    if (input_dtypes[i] == TRITONSERVER_TYPE_BYTES) {
      size_t max_size = Utils::GetMaxStringLen(reinterpret_cast<const unsigned char*>(input_buffer),
                                               buffer_byte_sizes[i]);
      max_str_sizes[input_names[i]] = max_size;
      size_t nif_size = max_size * input_shapes[i][0];

      std::unique_ptr<std::vector<wchar_t>> numpy_input_buffer(new std::vector<wchar_t>(nif_size, '\0'));
      Utils::ConstructNumpyStringArray(numpy_input_buffer->data(), static_cast<uint64_t>(max_size),
          reinterpret_cast<const unsigned char*>(input_buffer), buffer_byte_sizes[i]);
      input_buffers[i] = numpy_input_buffer->data();
      numpy_input_buffers.push_back(std::move(numpy_input_buffer));
    } else {
      input_buffers[i] = input_buffer;
    }

    if (input_memory_type == TRITONSERVER_MEMORY_GPU) {
      throw std::invalid_argument("Failed to get input buffer in CPU memory");
    }
  }

  const std::vector<std::string> & output_names = model_state->OutputNames();
  const std::vector<TRITONSERVER_DataType> & output_dtypes = model_state->OutputDtypes();

  py::gil_scoped_acquire l;
  instance_state->nvt.Transform(input_names, input_buffers, input_shapes,
          input_dtypes, max_str_sizes, output_names, output_dtypes, response);
}

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count) {
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void**>(&instance_state)));

  ModelState* model_state = instance_state->StateForModel();

  LOG(TRITONSERVER_LOG_INFO) << "model " << model_state->Name() << ", instance " <<
    instance_state->Name() << ", executing " << request_count << " requests";

  bool supports_batching = false;
  RETURN_IF_ERROR(model_state->SupportsFirstDimBatching(&supports_batching));

  uint64_t min_exec_start_ns = std::numeric_limits<uint64_t>::max();
  uint64_t max_exec_end_ns = 0;
  uint64_t total_batch_size = 0;
  std::string error = "";

  for (uint32_t r = 0; r < request_count; ++r) {
    uint64_t exec_start_ns = 0;
    SET_TIMESTAMP(exec_start_ns);
    min_exec_start_ns = std::min(min_exec_start_ns, exec_start_ns);

    auto request = requests[r];
    TRITONBACKEND_Response* response = NULL;
    auto err = TRITONBACKEND_ResponseNew(&response, request);
    LOG_IF_ERROR(err, "Failed to create response object");
    if (err) continue;

    try {
      transform_request(instance_state, request, response);
    } catch (const TritonException & e) {
      LOG_IF_ERROR(TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, e.error),
                  "Failed to send error response");
    } catch (const std::exception & e) {
      auto err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.what());
      LOG_IF_ERROR(TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
                  "Failed to send error response");
      TRITONSERVER_ErrorDelete(err);
    }

    /* TODO:
    if (supports_batching && (input_dims_counts[0] > 0)) {
      total_batch_size += input_shapes[0][0];
    } else {
      total_batch_size++;
    }
    */

    LOG_IF_ERROR(
      TRITONBACKEND_ResponseSend(
        response, TRITONSERVER_RESPONSE_COMPLETE_FINAL,
        nullptr /* success */),
        "failed sending response");

    uint64_t exec_end_ns = 0;
    SET_TIMESTAMP(exec_end_ns);
    max_exec_end_ns = std::max(max_exec_end_ns, exec_end_ns);

    LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportStatistics(
        instance_state->TritonModelInstance(), request, true /* success */,
        exec_start_ns, exec_start_ns, exec_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
        instance_state->TritonModelInstance(), total_batch_size,
        min_exec_start_ns, min_exec_start_ns, max_exec_end_ns,
        max_exec_end_ns),
        "failed reporting batch request statistics");

    LOG_IF_ERROR(
      TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }
  return nullptr;  // success
}

}  // extern "C"
}  // namespace nvtabular
}  // namespace backend
}  // namespace triton
