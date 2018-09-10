cmake_minimum_required(VERSION 3.8)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(ENABLE_NVTOOLS "Enable nvvp profiling of CPU code" OFF)
option(ENABLE_NO_ALIGNED_ALLOC "Enable aligned_alloc() function implemented in cuTT" OFF)

# find cuda
FIND_PACKAGE(CUDA REQUIRED)
message(STATUS "CUDA version:      ${CUDA_VERSION_STRING}")
message(STATUS "CUDA Include Path: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Libraries:    ${CUDA_LIBRARIES}")

list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

if(ENABLE_NVTOOLS)
    add_definitions(-DENABLE_NVTOOLS)
    list(APPEND CUDA_NVCC_FLAGS -lnvToolsExt)
endif()

message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

include(CheckFunctionExists)

# if to use cuTT aligned_alloc
if(ENABLE_NO_ALIGNED_ALLOC)
    add_definitions(-DNO_ALIGNED_ALLOC)
else()
    CHECK_FUNCTION_EXISTS(alligned_alloc HAVE_ALLIGNED_ALLOC)
    if(NOT HAVE_ALLIGNED_ALLOC)
        add_definitions(-DNO_ALIGNED_ALLOC)
    endif()
endif()


add_subdirectory(src)

