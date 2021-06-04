#!/bin/bash
set -e

# get the nvtabular_triton_backend directory
ci_directory="$(dirname -- "$(readlink -f -- "$0")")"
nvt_directory="$(dirname -- $ci_directory)"
cd $nvt_directory

# install dependencies
conda install -y -c conda-forge pybind11 rapidjson cmake cpplint

# check code
cpplint --linelength=120  --filter=-build/c++11,-build/include_subdir --root=src ./src/*

# install
echo "Installing NVTabular Triton Backend"
mkdir build && cd build && cmake .. && make -j
