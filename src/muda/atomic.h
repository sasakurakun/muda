#pragma once
#include <device_atomic_functions.h>
namespace muda
{
template <typename T>
__forceinline__ __device__ T atomic_cas(T* address, T compare, T val)
{
    return atomicCAS(address, compare, val);
}

template <typename T>
__forceinline__ __device__ T atomic_add(T* address, T val)
{
    return atomicAdd(address, val);
}

template <>
__forceinline__ __device__ double atomic_add<double>(double* address, double val)
{
    unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
    unsigned long long  old            = *address_as_ull;
    unsigned long long  assumed;

    do
    {
        assumed = old;
        old     = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while(assumed != old);

    return __longlong_as_double(old);
}

template <typename T>
__forceinline__ __device__ T atomic_sub(T* address, T val)
{
    return atomicSub(address, val);
}

template <typename T>
__forceinline__ __device__ T atomic_exch(T* address, T val)
{
    return atomicExch(address, val);
}

template <typename T>
__forceinline__ __device__ T atomic_max(T* address, T val)
{
    return atomicMax(address, val);
}

template <typename T>
__forceinline__ __device__ T atomic_min(T* address, T val)
{
    return atomicMin(address, val);
}

template <typename T>
__forceinline__ __device__ T atomic_and(T* address, T val)
{
    return atomicAnd(address, val);
}

template <typename T>
__forceinline__ __device__ T atomic_or(T* address, T val)
{
    return atomicOr(address, val);
}

template <typename T>
__forceinline__ __device__ T atomic_xor(T* address, T val)
{
    return atomicXor(address, val);
}
}  // namespace muda