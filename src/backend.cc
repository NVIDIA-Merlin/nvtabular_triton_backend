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
#include "model_state.hpp"
#include "model_inst_state.hpp"
#include "triton_utils.hpp"
#include "triton_python_backend_utils.hpp"

namespace triton { namespace backend { namespace nvtabular {

// This defines a python backend that creates an embedded interpreter with pybind11.
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

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

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

  try {
    py::initialize_interpreter();

    // we need to import our embedded extension module here
    py::module_::import("triton_python_backend_utils");

    // we have to manually release the GIL here otherwise we'll deadlock in future threads
    // but lets save the threadstate for shutdown
    auto thread_state = PyEval_SaveThread();
    check_triton(TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(thread_state)));
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Python interpreter is initialized");
  } catch (const std::exception & e) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, e.what());
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.what());
  }
  return nullptr;  // success
}

// Clean up the python interpreter when this backend is unloaded
TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend) {
  try {
    if (Py_IsInitialized()) {
      // acquire the GIL from the saved python thread state. Note that we're not using the
      // pybind gil_scoped_acquire object here, because releasing the GIL once it goes out of
      // scope can cause issues once the interpreter has been shutdown
      PyThreadState * thread_state;
      check_triton(TRITONBACKEND_BackendState(backend, reinterpret_cast<void**>(&thread_state)));
      PyEval_RestoreThread(thread_state);
      py::finalize_interpreter();
    }
    return nullptr;
  } catch (const std::exception & e) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, e.what());
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.what());
  }
}

// Initialize the ModelState, which stores global config across devices
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) {
  try {
    ModelState* model_state = new ModelState(model);
    return TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state));
  } catch (const TritonException & e) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, e.what());
    return e.error;
  } catch (const std::exception & e) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, e.what());
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.what());
  }
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
  ModelState * model_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, reinterpret_cast<void**>(&model_state)));
  LOG(TRITONSERVER_LOG_INFO) << "TRITONBACKEND_ModelFinalize: delete model state";
  delete model_state;
  return nullptr;  // success
}

// Initializes the ModelInstanceState, which wraps the python model we're exposing here
// We load up a new python model per ModelInstanceState because they can be hosted on
// different GPU's, and there might be GPU specific data for the model.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance) {
  try {
    ModelInstanceState* instance_state = new ModelInstanceState(instance);
    return TRITONBACKEND_ModelInstanceSetState(instance, reinterpret_cast<void*>(instance_state));
  } catch (const TritonException & e) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, e.what());
    return e.error;
  } catch (const std::exception & e) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, e.what());
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.what());
  }
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance) {
  LOG(TRITONSERVER_LOG_INFO) << "TRITONBACKEND_ModelInstanceFinalize: delete instance state";
  ModelInstanceState * instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void**>(&instance_state)));
  delete instance_state;
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
                                  const uint32_t request_count) {
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void**>(&instance_state)));

  std::vector<TRITONBACKEND_Response *> responses;
  for (uint32_t i = 0; i < request_count; ++i) {
    TRITONBACKEND_Response * response = NULL;
    LOG_IF_ERROR(TRITONBACKEND_ResponseNew(&response, requests[i]), "Failed to create response");
    responses.push_back(response);
  }

  try {
    instance_state->transform_requests(requests, responses.data(), request_count);
    return nullptr;
  } catch (const std::exception & e) {
    // all requests failed (possibly a bug in the python model code). return errors for each
    // request and cleanup
    LOG(TRITONSERVER_LOG_ERROR) << "Exception during transform_requests '" << e.what() << "'";
    auto err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.what());

    for (uint32_t i = 0; i < request_count; ++i) {
      if (responses[i]) {
        LOG_IF_ERROR(TRITONBACKEND_ResponseSend(responses[i], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
                    "Failed to send error response");
      }
      LOG_IF_ERROR(TRITONBACKEND_RequestRelease(requests[i], TRITONSERVER_REQUEST_RELEASE_ALL),
                  "Failed to release request");
    }

    // Note: we're purposefully not returning an Err here. (Doing so seems to segfault the
    // tritonserver process when it tries to respond the error  UNLESS we also don't release
    // the request (which means that on shutting down tritonserver is slow / maybe leaks memory ))
    // Since we've already sent an error response w/ ResponseSend this seems to be ok
    return nullptr;
  }
}
}  // extern "C"
}  // namespace nvtabular
}  // namespace backend
}  // namespace triton
