
set(CUTT_SOURCE_FILES
    calls.h
    CudaMem.cpp
    CudaMem.h
    CudaMemcpy.cu
    CudaMemcpy.h
    CudaUtils.cu
    CudaUtils.h
    cutt.cpp
    cutt.h
    cuttGpuModel.cpp
    cuttGpuModel.h
    cuttGpuModelKernel.cu
    cuttGpuModelKernel.h
    cuttkernel.cu
    cuttkernel.h
    cuttplan.h
    cuttplan.cpp
    cuttTimer.cpp
    cuttTimer.h
    cuttTypes.h
    int_vector.h
    TensorTester.cu
    TensorTester.h
    LRUCache.h)


add_library(cutt ${CUTT_SOURCE_FILES})
if(ENABLE_UMPIRE)
    target_link_libraries(cutt umpire)
endif()

# install library and include
install(TARGETS cutt
        ARCHIVE
            DESTINATION lib
        LIBRARY
            DESTINATION lib
        )
install(FILES cutt.h
        DESTINATION include
        )

add_executable(cutt_bench cutt_bench.cpp)
target_link_libraries(cutt_bench PUBLIC cutt)

add_executable(cutt_test cutt_test.cpp)
target_link_libraries(cutt_test PUBLIC cutt)


