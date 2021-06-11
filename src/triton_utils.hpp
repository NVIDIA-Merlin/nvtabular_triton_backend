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

#ifndef TRITON_UTILS_HPP_
#define TRITON_UTILS_HPP_

#include <string>

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

// TODO(benfred): zero cost log abstraction here (ShouldLog(Level) && TritonLogMessage(...))
#define LOG(severity) \
  TritonLogMessage(severity, __FILE__, __LINE__)

}  // namespace nvtabular
}  // namespace backend
}  // namespace triton
#endif  // TRITON_UTILS_HPP_
