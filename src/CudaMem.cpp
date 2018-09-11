
#include "CudaMem.h"

#ifdef HAVE_UMPIRE
#include <umpire/Umpire.hpp>
#include <umpire/strategy/DynamicPool.hpp>
#include <umpire/strategy/ThreadSafeAllocator.hpp>
#endif


// global allocator
#ifdef HAVE_UMPIRE

// set the allocator
auto& rm = umpire::ResourceManager::getInstance();
auto dynamic_pool = rm.makeAllocator<umpire::strategy::DynamicPool>(
        "GPUDynamicPool", rm.getAllocator("DEVICE"));
umpire::Allocator ts_dynamic_pool = rm.makeAllocator<umpire::strategy::ThreadSafeAllocator>(
        "ThreadSafeUMDynamicPool", rm.getAllocator("GPUDynamicPool"));


#endif

//----------------------------------------------------------------------------------------
//
// Allocate gpu memory
// pp = memory pointer
// len = length of the array
//
void allocate_device_T(void **pp, const size_t len, const size_t sizeofT) {
#ifdef HAVE_UMPIRE
  *pp = ts_dynamic_pool.allocate(sizeofT*len);
#else
  cudaCheck(cudaMalloc(pp, sizeofT*len));
#endif
}

//----------------------------------------------------------------------------------------
//
// Deallocate gpu memory
// pp = memory pointer
//
void deallocate_device_T(void **pp) {
#ifdef HAVE_UMPIRE
  ts_dynamic_pool.deallocate((void *) (*pp) );
#else
  if (*pp != NULL) {
    cudaCheck(cudaFree((void *)(*pp)));
    *pp = NULL;
  }
#endif
}