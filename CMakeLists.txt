cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(semantic_meshes LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CUDA_ARCHITECTURES OFF)

add_subdirectory(extern)
add_subdirectory(src)

option(BUILD_PYTHON_INTERFACE "Build python interface" ON)
if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()
