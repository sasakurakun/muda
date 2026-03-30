// don't place #pragma once at the beginning of this file
// because it should be inserted in multiple files

#define MUDA_CUB_WRAPPER_IMPL(x)                                               \
    cudaStream_t _stream            = this->stream();                          \
    size_t       temp_storage_bytes = 0;                                       \
    void*        d_temp_storage     = nullptr;                                 \
                                                                               \
    checkCudaErrors(x);                                                        \
                                                                               \
    MUDA_ASSERT(temp_storage_bytes < (size_t(1) << 34),                         \
                "CUB temp_storage_bytes looks overflowed: %s, temp_storage_bytes=%zu", \
                #x,                                                            \
                temp_storage_bytes);                                            \
                                                                               \
    d_temp_storage = (void*)prepare_buffer(temp_storage_bytes);                \
                                                                               \
    checkCudaErrors(x);                                                        \
                                                                               \
    return *this;

#define MUDA_CUB_WRAPPER_FOR_COMPUTE_GRAPH_IMPL(x)                                                        \
    std::string_view name{__func__};                                                                      \
    ComputeGraphBuilder::invoke_phase_actions(                                                            \
        [&]                                                                                               \
        {                                                                                                 \
            cudaStream_t _stream = this->stream();                                                        \
            checkCudaErrors(x);                                                                           \
        },                                                                                                \
        [&]                                                                                               \
        {                                                                                                 \
            MUDA_ASSERT(!ComputeGraphBuilder::is_building() || d_temp_storage != nullptr,                 \
                        "d_temp_storage must not be nullptr when building graph. you should not"          \
                        "query the temp_storage_size when building a compute graph, please do it outside" \
                        "a compute graph.");                                                              \
            ComputeGraphBuilder::capture(                                                                 \
                name, [&](cudaStream_t _stream) { checkCudaErrors(x); });                                 \
        });                                                                                               \
    return *this;
