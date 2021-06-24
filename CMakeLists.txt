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

cmake_minimum_required(VERSION 3.17)

project(tritonnvtabularbackend LANGUAGES C CXX)

find_package(pybind11 REQUIRED)

#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
# GPU support is disabled by default because nvtabular backend doesn't
# support GPUs.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(CMAKE_BUILD_TYPE Release)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
# FetchContent's composibility isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_nvtabular.ldscript libtriton_nvtabular.ldscript COPYONLY)

add_library(
  triton-nvtabular-backend SHARED
  src/backend.cc
)

add_library(
  TritonNVTabularBackend::triton-nvtabular-backend ALIAS triton-nvtabular-backend
)

target_include_directories(
  triton-nvtabular-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    
)

target_compile_features(triton-nvtabular-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-nvtabular-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Werror -Wno-unused-parameter -Wno-type-limits>
)

target_link_libraries(
  triton-nvtabular-backend
  PRIVATE
    triton-backend-utils    # from repo-backend
    triton-core-serverstub  # from repo-core
    pybind11::embed
)

set_target_properties(
  triton-nvtabular-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_nvtabular
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_nvtabular.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_nvtabular.ldscript"
)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonNVTabularBackend)

install(
  TARGETS
    triton-nvtabular-backend
  EXPORT
    triton-nvtabular-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/nvtabular
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/nvtabular
)

install(
  EXPORT
    triton-nvtabular-backend-targets
  FILE
    TritonNVTabularBackendTargets.cmake
  NAMESPACE
    TritonNVTabularBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonNVTabularBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonNVTabularBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonNVTabularBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-nvtabular-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonNVTabularBackendTargets.cmake
  NAMESPACE TritonNVTabularBackend::
)

export(PACKAGE TritonNVTabularBackend)
