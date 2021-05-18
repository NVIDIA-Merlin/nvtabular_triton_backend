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

#ifndef PYBIND_H_
#define PYBIND_H_

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace triton {
namespace backend {
namespace nvtabular {

class Pybind {
public:
  static Pybind *getInstance() {
    if (inst_ == NULL) {
      inst_ = new Pybind();
    }
    return (inst_);
  }

  void InitPythonInterpreter() {
    if (!inter_started) {
      py::initialize_interpreter();
      inter_started = true;
      inter_stopped = false;
    }
  }

  void FinalizePythonInterpreter() {
    if (!inter_stopped) {
      py::finalize_interpreter();
      inter_stopped = true;
      inter_started = false;
    }
  }

private:
  static Pybind *inst_;
  Pybind() : inter_started(false), inter_stopped(true) {}
  Pybind(const Pybind &);
  Pybind &operator=(const Pybind &);
  bool inter_started;
  bool inter_stopped;
};

Pybind *Pybind::inst_ = NULL;

} // namespace nvtabular
} // namespace backend
} // namespace triton

#endif /* PYBIND_H_ */
