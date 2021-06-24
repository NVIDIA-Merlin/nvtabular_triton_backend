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

    const char *model_name;
    check_triton(TRITONBACKEND_ModelName(triton_model, &model_name));
    name_ = model_name;

    // get the python module name to use from the model config
    common::TritonJson::Value model_json;
    check_triton(model_json.Parse(buffer, byte_size));
    common::TritonJson::Value parameters;
    if (model_json.Find("parameters", &parameters)) {
      common::TritonJson::Value python_module_json;
      if (parameters.Find("python_module", &python_module_json)) {
        check_triton(python_module_json.MemberAsString("string_value", &python_module_));
      }
    }

    check_triton(TRITONBACKEND_ModelVersion(triton_model, &version_));
    check_triton(TRITONBACKEND_ModelServer(triton_model, &triton_server_));

    TRITONBACKEND_ArtifactType artifact_type;
    const char *path;
    check_triton(TRITONBACKEND_ModelRepository(triton_model, &artifact_type, &path));
    path_ = path;
    check_triton(TRITONSERVER_MessageDelete(config_message));
  }

  // Get the handle to the TRITONBACKEND model.
  TRITONBACKEND_Model *TritonModel() { return triton_model_; }

  // Get the name and version of the model.
  const std::string & Name() const { return name_; }
  uint64_t Version() const { return version_; }
  const std::string & Path() const { return path_; }
  const std::string & ModelConfig() { return model_config_; }
  const std::string & PythonModule() { return python_module_; }

 private:
  TRITONSERVER_Server *triton_server_;
  TRITONBACKEND_Model *triton_model_;
  std::string name_;
  uint64_t version_;
  std::string path_;
  std::string model_config_;
  std::string python_module_;
};

}  // namespace nvtabular
}  // namespace backend
}  // namespace triton

#endif  // MODEL_STATE_HPP_
