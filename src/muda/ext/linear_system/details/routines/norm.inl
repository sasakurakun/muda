#include <muda/ext/linear_system/type_mapper/data_type_mapper.h>
#include <type_traits>
namespace muda
{
namespace details::linear_system
{
    template <typename T>
    MUDA_INLINE void norm_common_check(CDenseVectorView<T> x)
    {
        MUDA_ASSERT(x.data(), "Vector x is empty");
    }
}  // namespace details::linear_system

template <typename T>
T LinearSystemContext::norm(CDenseVectorView<T> x)
{
    T result;
    norm(x, &result);
    sync();
    return result;
}

template <typename T>
void LinearSystemContext::norm(CDenseVectorView<T> x, VarView<T> result)
{
    set_pointer_mode_device();
    details::linear_system::norm_common_check(x);
    auto size = x.size() / x.inc();
    if constexpr(std::is_same_v<T, float>)
    {
        checkCudaErrors(cublasSnrm2(cublas(), size, x.data(), x.inc(), result.data()));
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        checkCudaErrors(cublasDnrm2(cublas(), size, x.data(), x.inc(), result.data()));
    }
    else
    {
        auto type = cuda_data_type<T>();
        checkCudaErrors(cublasNrm2Ex(
            cublas(), size, x.data(), type, x.inc(), result.data(), type, type));
    }
}

template <typename T>
void LinearSystemContext::norm(CDenseVectorView<T> x, T* result)
{
    set_pointer_mode_host();
    details::linear_system::norm_common_check(x);
    auto size = x.size() / x.inc();
    if constexpr(std::is_same_v<T, float>)
    {
        checkCudaErrors(cublasSnrm2(cublas(), size, x.data(), x.inc(), result));
    }
    else if constexpr(std::is_same_v<T, double>)
    {
        checkCudaErrors(cublasDnrm2(cublas(), size, x.data(), x.inc(), result));
    }
    else
    {
        auto type = cuda_data_type<T>();
        checkCudaErrors(
            cublasNrm2Ex(cublas(), size, x.data(), type, x.inc(), result, type, type));
    }
}
}  // namespace muda