cmake_minimum_required(VERSION 3.8.2)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
include(CheckFunctionExists)

option(ENABLE_NVTOOLS "Enable nvvp profiling of CPU code" OFF)
option(ENABLE_NO_ALIGNED_ALLOC "Enable aligned_alloc() function implemented in cuTT" OFF)
option(ENABLE_UMPIRE "Enable umpire for memory management" OFF)

set(CUDA_HOST_COMPILER ${CMAKE_CUDA_HOST_COMPILER})

# find cuda
FIND_PACKAGE(CUDA REQUIRED)
message(STATUS "CUDA version:      ${CUDA_VERSION_STRING}")
message(STATUS "CUDA Include Path: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Libraries:    ${CUDA_LIBRARIES}")
message(STATUS "CUDA CXX Compiler:    ${CUDA_HOST_COMPILER}")


message(STATUS "Original CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
list(FIND CUDA_NVCC_FLAGS "--std" CUDA_STANDARD_FLAG)
if(CUDA_STANDARD_FLAG EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS "--std;c++11")
endif()

# ENABLE_NVTOOLS
if(ENABLE_NVTOOLS)
    add_definitions(-DENABLE_NVTOOLS)
    list(APPEND CUDA_NVCC_FLAGS -lnvToolsExt)
endif()

message(STATUS "Current CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")


# ENABLE_NO_ALIGNED_ALLOC
if(ENABLE_NO_ALIGNED_ALLOC)
    add_definitions(-DNO_ALIGNED_ALLOC)
else()
    CHECK_FUNCTION_EXISTS(alligned_alloc HAVE_ALLIGNED_ALLOC)
    if(NOT HAVE_ALLIGNED_ALLOC)
        add_definitions(-DNO_ALIGNED_ALLOC)
    endif()
endif()

# ENABLE_UMPIRE
if (ENABLE_UMPIRE)
    set(umpire_DIR ${UMPIRE_INSTALL_DIR}/share/umpire/cmake)
    find_package(umpire REQUIRED)
    add_definitions(-DHAVE_UMPIRE)
endif ()

add_subdirectory(src)

