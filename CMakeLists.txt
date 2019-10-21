cmake_minimum_required(VERSION 3.10)

# enable CXX
enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# enable cuda
enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

message(STATUS "CUDA Include Dirs: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
message(STATUS "CUDA Host Compiler: " ${CMAKE_CUDA_HOST_COMPILER})

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

option(ENABLE_NVTOOLS "Enable nvvp profiling of CPU code" OFF)
option(ENABLE_NO_ALIGNED_ALLOC "Enable aligned_alloc() function implemented in cuTT" OFF)
option(ENABLE_UMPIRE "Enable umpire for memory management" OFF)
include(CheckFunctionExists)



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
    # this is available in C11 N.B. std::aligned_alloc is available in C++17
    CHECK_FUNCTION_EXISTS(aligned_alloc HAVE_ALIGNED_ALLOC)
    if(NOT HAVE_ALIGNED_ALLOC)
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

