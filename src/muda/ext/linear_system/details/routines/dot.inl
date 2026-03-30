namespace muda
{
namespace details::linear_system
{
    template <typename T>
    MUDA_INLINE T host_dot_fallback(CDenseVectorView<T> x, CDenseVectorView<T> y)
    {
        std::vector<T> hx(x.size());
        std::vector<T> hy(y.size());
        checkCudaErrors(cudaMemcpy(hx.data(),
                                   x.data(),
                                   sizeof(T) * x.size(),
                                   cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hy.data(),
                                   y.data(),
                                   sizeof(T) * y.size(),
                                   cudaMemcpyDeviceToHost));

        const auto n = x.size() / x.inc();
        T          s = static_cast<T>(0);
        for(int i = 0; i < n; ++i)
        {
            s += hx[static_cast<size_t>(i) * x.inc()]
                 * hy[static_cast<size_t>(i) * y.inc()];
        }
        return s;
    }

    template <typename T>
    MUDA_INLINE void dot_common_check(CDenseVectorView<T> x, CDenseVectorView<T> y)
    {
        MUDA_ASSERT(x.data() && y.data(), "x.data() and y.data() should not be nullptr");
        MUDA_ASSERT(x.size() / x.inc() == y.size() / y.inc(),
                    "x (size=%lld, inc=%d) should be the same as y (size=%lld, inc=%d)",
                    x.size(),
                    x.inc(),
                    y.size(),
                    y.inc());
    }
}  // namespace details::linear_system


template <typename T>
void LinearSystemContext::dot(CDenseVectorView<T> x, CDenseVectorView<T> y, T* result)
{
    set_pointer_mode_host();
    details::linear_system::dot_common_check(x, y);

    auto type = cuda_data_type<T>();
    auto size = x.size() / x.inc();

    auto status = cublasDotEx(
        cublas(), size, x.data(), type, x.inc(), y.data(), type, y.inc(), result, type, type);
    if(status == CUBLAS_STATUS_NOT_SUPPORTED)
    {
        if constexpr(std::is_same_v<T, float>)
        {
            checkCudaErrors(cublasSdot(cublas(), size, x.data(), x.inc(), y.data(), y.inc(), result));
            return;
        }
        else if constexpr(std::is_same_v<T, double>)
        {
            auto dstatus = cublasDdot(cublas(), size, x.data(), x.inc(), y.data(), y.inc(), result);
            if(dstatus == CUBLAS_STATUS_SUCCESS)
                return;
        }

        *result = details::linear_system::host_dot_fallback(x, y);
        return;
    }
    checkCudaErrors(status);
}

template <typename T>
T LinearSystemContext::dot(CDenseVectorView<T> x, CDenseVectorView<T> y)
{
    T result;
    dot(x, y, &result);
    sync();
    return result;
}

template <typename T>
void LinearSystemContext::dot(CDenseVectorView<T> x, CDenseVectorView<T> y, VarView<T> result)
{
    set_pointer_mode_device();
    details::linear_system::dot_common_check(x, y);

    auto type = cuda_data_type<T>();
    auto size = x.size() / x.inc();


    auto status = cublasDotEx(
        cublas(), size, x.data(), type, x.inc(), y.data(), type, y.inc(), result.data(), type, type);
    if(status == CUBLAS_STATUS_NOT_SUPPORTED)
    {
        if constexpr(std::is_same_v<T, float>)
        {
            checkCudaErrors(cublasSdot(
                cublas(), size, x.data(), x.inc(), y.data(), y.inc(), result.data()));
            return;
        }
        else if constexpr(std::is_same_v<T, double>)
        {
            auto dstatus =
                cublasDdot(cublas(), size, x.data(), x.inc(), y.data(), y.inc(), result.data());
            if(dstatus == CUBLAS_STATUS_SUCCESS)
                return;
        }

        auto host_result = details::linear_system::host_dot_fallback(x, y);
        checkCudaErrors(cudaMemcpy(result.data(),
                                   &host_result,
                                   sizeof(T),
                                   cudaMemcpyHostToDevice));
        return;
    }
    checkCudaErrors(status);
}

}  // namespace muda