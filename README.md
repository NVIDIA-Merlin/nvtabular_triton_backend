<!--
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
-->

[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](https://opensource.org/licenses/Apache-2.0)

# NVTabular C++ Triton Backend
This repo includes the source code for the NVTabular C++ Triton Backend.

`nvcr.io/nvstaging/merlin/merlin-inference` container includes all the required
packages and libraries to build the C++ backend from source. Please follow the
steps below;

```
$ conda install -c conda-forge rapidjson pybind11 cmake
$ git clone https://github.com/NVIDIA-Merlin/nvtabular_triton_backend.git
$ cd nvtabular_triton_backend/
$ mkdir build
$ cd build
$ cmake ..
$ make install
$ mkdir /opt/tritonserver/nvtabular
$ cp libtriton_nvtabular.so /opt/tritonserver/nvtabular/
```

Before start serving a model with nvtabular backend, run the following command;

```
$ export LD_LIBRARY_PATH=/conda/envs/merlin/lib/:$LD_LIBRARY_PATH
```

If the path of the libraries (i.e. python libs) has already been added to the `LD_LIBRARY_PATH`,
you don't have to run the command above.

The model that will use NVTabular C++ backend should have the following in the
config.pbtxt file

```
backend: "nvtabular"
```

Note that the following required Triton repositories will be pulled and used in
the build. By default the "main" branch/tag will be used for each repo
but the listed CMake argument can be used to override.

* triton-inference-server/backend: -DTRITON_BACKEND_REPO_TAG=[tag]
* triton-inference-server/core: -DTRITON_CORE_REPO_TAG=[tag]
* triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=[tag]
