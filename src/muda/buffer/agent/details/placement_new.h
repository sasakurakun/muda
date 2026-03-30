#pragma once

#include <cstddef>

#if defined(UIPC_COREX_CUDA10_COMPAT)
// Corex clang-CUDA: device placement-new must be declared for kernels that use
// `new(ptr) T`, but defining it in this header caused ODR violations when both
// kernel_construct.inl and kernel_copy_construct.inl appear in one TU.
// Single non-inline definition: corex_device_placement_new.cu (CUDA backend).
__device__ void* operator new(std::size_t, void* p) noexcept;
__device__ void  operator delete(void*, void*) noexcept;
#endif
