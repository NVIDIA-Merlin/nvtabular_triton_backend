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

#ifndef MODEL_STATE_HPP_
#define MODEL_STATE_HPP_

#include <string>
#include <utility>
#include <vector>

#include "triton/backend/backend_common.h"
#include "triton_utils.hpp"


namespace triton {
namespace backend {
namespace nvtabular {

class ModelState {
 public:
  explicit ModelState(TRITONBACKEND_Model *triton_model)
    : triton_model_(triton_model) {
    TRITONSERVER_Message *config_message;
    check_triton(TRITONBACKEND_ModelConfig(triton_model, 1, &config_message));

    const char *buffer;
    size_t byte_size;
    check_triton(TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));
    model_config_.assign(buffer, byte_size);
    check_triton(TRITONSERVER_MessageDelete(config_message));

    const char *model_name;
    check_triton(TRITONBACKEND_ModelName(triton_model, &model_name));
    name_ = model_name;

    check_triton(TRITONBACKEND_ModelVersion(triton_model, &version_));
    check_triton(TRITONBACKEND_ModelServer(triton_model, &triton_server_));

    TRITONBACKEND_ArtifactType artifact_type;
    const char *path;
    check_triton(TRITONBACKEND_ModelRepository(triton_model, &artifact_type, &path));
    path_ = path;
  }

  // Get the handle to the TRITONBACKEND model.
  TRITONBACKEND_Model *TritonModel() { return triton_model_; }

  // Get the name and version of the model.
  const std::string & Name() const { return name_; }
  uint64_t Version() const { return version_; }
  const std::string & Path() const { return path_; }
  const std::string & ModelConfig() { return model_config_; }

 private:
  TRITONSERVER_Server *triton_server_;
  TRITONBACKEND_Model *triton_model_;
  std::string name_;
  uint64_t version_;
  std::string path_;
  std::string model_config_;
};

}  // namespace nvtabular
}  // namespace backend
}  // namespace triton

#endif  // MODEL_STATE_HPP_
