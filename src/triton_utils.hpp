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

#ifndef TRITON_UTILS_HPP_
#define TRITON_UTILS_HPP_

#include <exception>
#include <string>
#include <vector>

namespace triton {
namespace backend {
namespace nvtabular {

struct TritonException : public std::exception {
  TRITONSERVER_Error * error;
  explicit TritonException(TRITONSERVER_Error * error) : error(error) {}

  const char * what() const noexcept override {
    return TRITONSERVER_ErrorMessage(error);
  }

  ~TritonException() {
    if (error) {
       TRITONSERVER_ErrorDelete(error);
    }
  }
};

static inline void check_triton(TRITONSERVER_Error * e) {
  if (e) {
    throw TritonException(e);
  }
}

// Adaptor to log triton messages using a c++ stringstream to compose messages
struct TritonLogMessage {
 public:
  TritonLogMessage(TRITONSERVER_LogLevel level, const char* filename, const int line)
    : level(level), filename(filename), line(line) {
  }

  template <typename T>
  TritonLogMessage & operator << (T t) {
    message << t;
    return *this;
  }

  ~TritonLogMessage() {
    // logging a message if you fail to log a message seems doomed to fail
    // but this matches the behaviour of LOG_MESSAGE
    LOG_IF_ERROR(TRITONSERVER_LogMessage(level, filename, line, message.str().c_str()),
                 "Failed to log message");
  }

  TRITONSERVER_LogLevel level;
  const char* filename;
  const int line;
  std::stringstream message;
};


class Input {
 public:
  Input() {}

  explicit Input(TRITONBACKEND_Input * triton_input) {
    uint32_t buffer_count;
    check_triton(TRITONBACKEND_InputProperties(triton_input,
      &name, &dtype, &shape, &dims, NULL, &buffer_count));

    if (buffer_count != 1) {
      std::stringstream err;
      err << "buffer_count " << buffer_count << " not supported for input '" << name << "'";
      throw std::invalid_argument(err.str());
    }

    int64_t memory_type_id;
    check_triton(TRITONBACKEND_InputBuffer(triton_input, 0, &buffer, &buffer_size, &memory_type, &memory_type_id));
  }

  ~Input() {}

  Input(Input &&) = default;
  Input& operator=(Input && other) = default;
  Input(const Input &) = delete;
  Input& operator=(const Input &) = delete;

  // core information about this tensor
  TRITONSERVER_DataType dtype;
  TRITONSERVER_MemoryType memory_type;
  const char * name = NULL;
  uint32_t dims;
  const int64_t * shape = NULL;
  const void * buffer = NULL;
  uint64_t buffer_size;
};


class InferenceRequest {
 public:
  InferenceRequest() {}

  explicit InferenceRequest(TRITONBACKEND_Request * request) {
    check_triton(TRITONBACKEND_RequestId(request, &request_id));
    check_triton(TRITONBACKEND_RequestCorrelationId(request, &correlation_id));
    check_triton(TRITONBACKEND_RequestInputCount(request, &input_count));
    check_triton(TRITONBACKEND_RequestOutputCount(request, &output_count));

    for (uint32_t i = 0; i < input_count; ++i) {
      TRITONBACKEND_Input * input;
      check_triton(TRITONBACKEND_RequestInputByIndex(request, i, &input));
      inputs.push_back(Input(input));
    }
  }

  const std::vector<Input> * get_inputs() const { return &inputs; }
  uint64_t get_correlation_id() const { return correlation_id; }
  const char * get_request_id() const { return request_id; }

  InferenceRequest(InferenceRequest &&) = default;
  InferenceRequest& operator=(InferenceRequest && other) = default;
  InferenceRequest(const InferenceRequest &) = delete;
  InferenceRequest& operator=(const InferenceRequest &) = delete;

 protected:
  std::vector<Input> inputs;
  const char * request_id;
  uint64_t correlation_id;
  uint32_t input_count;
  uint32_t output_count;
};

static inline uint64_t timestamp_ns() {
  uint64_t ret;
  SET_TIMESTAMP(ret);
  return ret;
}

}  // namespace nvtabular
}  // namespace backend
}  // namespace triton
// TODO(benfred): zero cost log abstraction here (ShouldLog(Level) && TritonLogMessage(...))
#define LOG(severity) \
  TritonLogMessage(severity, __FILE__, __LINE__)

// macro to define a class / function as local to this backend
// (useful for quieting warnings around visibility with pybind objects)
// https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes
#define NVT_LOCAL __attribute__ ((visibility ("hidden")))
#endif  // TRITON_UTILS_HPP_
