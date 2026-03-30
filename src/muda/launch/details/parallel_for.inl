#include <muda/compute_graph/compute_graph.h>
#include <muda/type_traits/always.h>
#include <muda/launch/kernel_tag.h>
namespace muda
{
namespace details
{
    /*
    **************************************************************************
    * This part is the core of the "launch part of muda"                     *
    **************************************************************************
    * F: the callable object                                                 *
    * UserTag: the tag struct for user to recognize on profiling             *
    **************************************************************************
    */
    template <typename F, typename UserTag>
    MUDA_GLOBAL void parallel_for_kernel(ParallelForCallable<F> f)
    {
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        // Corex clang: if constexpr + std::is_invocable on device lambdas can miscompile
        // (e.g. invalid addrspacecast). Force the common (int) index path.
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto i   = tid;
        if(i < f.count)
            f.callable(i);
#else
        if constexpr(std::is_invocable_v<F, int>)
        {
            auto tid = blockIdx.x * blockDim.x + threadIdx.x;
            auto i   = tid;
            if(i < f.count)
            {
                f.callable(i);
            }
        }
        else if constexpr(std::is_invocable_v<F, ParallelForDetails>)
        {
            ParallelForDetails details{ParallelForType::DynamicBlocks,
                                       static_cast<int>(blockIdx.x * blockDim.x
                                                        + threadIdx.x),
                                       f.count};
            if(details.i() < details.total_num())
            {
                f.callable(details);
            }
        }
        else
        {
            // Some toolchains cannot evaluate device-callable lambdas in std::is_invocable.
            auto tid = blockIdx.x * blockDim.x + threadIdx.x;
            auto i   = tid;
            if(i < f.count)
            {
                f.callable(i);
            }
        }
#endif
    }

    template <typename F, typename UserTag>
    MUDA_GLOBAL void grid_stride_loop_kernel(ParallelForCallable<F> f)
    {
#if defined(UIPC_COREX_CUDA10_COMPAT) && UIPC_COREX_CUDA10_COMPAT
        auto tid       = blockIdx.x * blockDim.x + threadIdx.x;
        auto grid_size = gridDim.x * blockDim.x;
        auto i         = tid;
        for(; i < f.count; i += grid_size)
            f.callable(i);
#else
        if constexpr(std::is_invocable_v<F, int>)
        {
            auto tid       = blockIdx.x * blockDim.x + threadIdx.x;
            auto grid_size = gridDim.x * blockDim.x;
            auto i         = tid;
            for(; i < f.count; i += grid_size)
                f.callable(i);
        }
        else if constexpr(std::is_invocable_v<F, ParallelForDetails>)
        {
            auto tid        = blockIdx.x * blockDim.x + threadIdx.x;
            auto grid_size  = gridDim.x * blockDim.x;
            auto block_size = blockDim.x;
            auto i          = tid;
            auto count      = f.count;
            auto round      = (count + grid_size - 1) / grid_size;
            for(int j = 0; i < count; i += grid_size, ++j)
            {
                ParallelForDetails details{
                    ParallelForType::GridStrideLoop, static_cast<int>(i), count};

                details.m_total_batch = round;
                details.m_batch_i     = j;
                if(i + block_size > details.total_num())  // the block may be incomplete in the last round
                    details.m_active_num_in_block = count - j * grid_size;
                else
                    details.m_active_num_in_block = block_size;
                f.callable(details);
            }
        }
        else
        {
            // Some toolchains cannot evaluate device-callable lambdas in std::is_invocable.
            auto tid       = blockIdx.x * blockDim.x + threadIdx.x;
            auto grid_size = gridDim.x * blockDim.x;
            auto i         = tid;
            for(; i < f.count; i += grid_size)
                f.callable(i);
        }
#endif
    }
}  // namespace details


template <bool kGridStrideMode>
template <typename F, typename UserTag>
MUDA_HOST ParallelFor<kGridStrideMode>& ParallelFor<kGridStrideMode>::apply(int count, F&& f)
{
    if constexpr(COMPUTE_GRAPH_ON)
    {
        using CallableType = raw_type_t<F>;

        ComputeGraphBuilder::invoke_phase_actions(
            [&] {  // direct invoke
                invoke<F, UserTag>(count, std::forward<F>(f));
            },
            [&]
            {
                // as node parms
                auto parms = as_node_parms<F, UserTag>(count, std::forward<F>(f));
                details::ComputeGraphAccessor().set_kernel_node(parms);
            },
            [&]
            {
                // topo build
                details::ComputeGraphAccessor().set_kernel_node<details::ParallelForCallable<CallableType>>(
                    nullptr);
            });
    }
    else
    {
        invoke<F, UserTag>(count, std::forward<F>(f));
    }
    this->pop_kernel_label();
    return *this;
}

template <bool kGridStrideMode>
template <typename F, typename UserTag>
MUDA_HOST ParallelFor<kGridStrideMode>& ParallelFor<kGridStrideMode>::apply(int count, F&& f, Tag<UserTag>)
{
    return apply<F, UserTag>(count, std::forward<F>(f));
}

template <bool kGridStrideMode>
template <typename F, typename UserTag>
MUDA_NODISCARD MUDA_HOST auto ParallelFor<kGridStrideMode>::make_as_node_parms_dynamic(int count, F&& f)
    -> std::shared_ptr<NodeParms<F>>
{
    using CallableType = raw_type_t<F>;
    auto parms         = std::make_shared<NodeParms<F>>(std::forward<F>(f), count);
    m_block_dim        = calculate_block_dim<F, UserTag>(count);
    auto n_blocks      = calculate_grid_dim(count, m_block_dim);
    parms->func((void*)details::parallel_for_kernel<CallableType, UserTag>);
    parms->grid_dim(n_blocks);
    parms->block_dim(m_block_dim);
    parms->shared_mem_bytes(static_cast<uint32_t>(m_shared_mem_size));
    parms->parse([](details::ParallelForCallable<CallableType>& p) -> std::vector<void*>
                 { return {&p}; });
    return parms;
}

template <bool kGridStrideMode>
template <typename F, typename UserTag>
MUDA_NODISCARD MUDA_HOST auto ParallelFor<kGridStrideMode>::make_as_node_parms_grid_stride(int count, F&& f)
    -> std::shared_ptr<NodeParms<F>>
{
    using CallableType = raw_type_t<F>;
    auto parms = std::make_shared<NodeParms<F>>(std::forward<F>(f), count);
    parms->func((void*)details::grid_stride_loop_kernel<CallableType, UserTag>);
    parms->grid_dim(m_grid_dim);
    parms->block_dim(m_block_dim);
    parms->shared_mem_bytes(static_cast<uint32_t>(m_shared_mem_size));
    parms->parse([](details::ParallelForCallable<CallableType>& p) -> std::vector<void*>
                 { return {&p}; });
    return parms;
}

template <bool kGridStrideMode>
template <typename F, typename UserTag>
MUDA_HOST void ParallelFor<kGridStrideMode>::invoke_parallel_for_dynamic(int count, F&& f)
{
    using CallableType = raw_type_t<F>;
    m_block_dim        = calculate_block_dim<F, UserTag>(count);
    auto n_blocks      = calculate_grid_dim(count, m_block_dim);
    auto callable      = details::ParallelForCallable<CallableType>{f, count};
    details::parallel_for_kernel<CallableType, UserTag>
        <<<n_blocks, m_block_dim, m_shared_mem_size, this->m_stream>>>(callable);
}

template <bool kGridStrideMode>
template <typename F, typename UserTag>
MUDA_HOST void ParallelFor<kGridStrideMode>::invoke_parallel_for_grid_stride(int count, F&& f)
{
    using CallableType = raw_type_t<F>;
    auto callable = details::ParallelForCallable<CallableType>{f, count};
    details::grid_stride_loop_kernel<CallableType, UserTag>
        <<<m_grid_dim, m_block_dim, m_shared_mem_size, this->m_stream>>>(callable);
}

template <bool kGridStrideMode>
template <typename F, typename UserTag>
MUDA_NODISCARD MUDA_HOST auto ParallelFor<kGridStrideMode>::as_node_parms(int count, F&& f)
    -> std::shared_ptr<NodeParms<F>>
{
    if constexpr(kGridStrideMode)
        return make_as_node_parms_grid_stride<F, UserTag>(count, std::forward<F>(f));
    else
        return make_as_node_parms_dynamic<F, UserTag>(count, std::forward<F>(f));
}

template <bool kGridStrideMode>
template <typename F, typename UserTag>
MUDA_NODISCARD MUDA_HOST auto ParallelFor<kGridStrideMode>::as_node_parms(int count, F&& f, Tag<UserTag>)
    -> std::shared_ptr<NodeParms<F>>
{
    return as_node_parms<F, UserTag>(count, std::forward<F>(f));
}

template <bool kGridStrideMode>
template <typename F, typename UserTag>
MUDA_HOST void ParallelFor<kGridStrideMode>::invoke(int count, F&& f)
{
    // check_input(count);
    if(count > 0)
    {
        if constexpr(kGridStrideMode)
            invoke_parallel_for_grid_stride<F, UserTag>(count, std::forward<F>(f));
        else
            invoke_parallel_for_dynamic<F, UserTag>(count, std::forward<F>(f));
    }
}

template <bool kGridStrideMode>
template <typename F, typename UserTag>
MUDA_INLINE MUDA_GENERIC int ParallelFor<kGridStrideMode>::calculate_block_dim(int count) const MUDA_NOEXCEPT
{
    using CallableType  = raw_type_t<F>;
    int best_block_size = -1;
    if(m_block_dim <= 0)  // automatic choose
    {
        int min_grid_size = -1;
#ifdef __CUDA_ARCH__
        int cached_block_size = -1;
#else
        static thread_local int cached_block_size = -1;
#endif
        if(cached_block_size <= 0)
        {
            if constexpr(kGridStrideMode)
            {
                checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
                    &min_grid_size,
                    &cached_block_size,
                    details::grid_stride_loop_kernel<CallableType, UserTag>,
                    m_shared_mem_size));
            }
            else
            {
                checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
                    &min_grid_size,
                    &cached_block_size,
                    details::parallel_for_kernel<CallableType, UserTag>,
                    m_shared_mem_size));
            }
        }
        best_block_size = cached_block_size;
    }
    else
    {
        best_block_size = m_block_dim;
    }
    MUDA_ASSERT(best_block_size >= 0, "Invalid block dim");
    return best_block_size;
}

template <bool kGridStrideMode>
MUDA_INLINE MUDA_GENERIC int ParallelFor<kGridStrideMode>::calculate_grid_dim(int count) const MUDA_NOEXCEPT
{
    return calculate_grid_dim(count, m_grid_dim);
}

template <bool kGridStrideMode>
MUDA_INLINE MUDA_GENERIC int ParallelFor<kGridStrideMode>::calculate_grid_dim(int count, int block_dim) MUDA_NOEXCEPT
{
    auto min_threads = count;
    auto min_blocks  = (min_threads + block_dim - 1) / block_dim;
    return min_blocks;
}

template <bool kGridStrideMode>
MUDA_INLINE MUDA_GENERIC void ParallelFor<kGridStrideMode>::check_input(int count) const MUDA_NOEXCEPT
{
    MUDA_KERNEL_ASSERT(count >= 0, "count must be >= 0");
    MUDA_KERNEL_ASSERT(m_block_dim > 0, "blockDim must be > 0");
}

MUDA_INLINE MUDA_DEVICE int ParallelForDetails::active_num_in_block() const MUDA_NOEXCEPT
{
    if(m_type == ParallelForType::DynamicBlocks)
    {
        auto block_id = blockIdx.x;
        return (blockIdx.x == gridDim.x - 1) ? m_total_num - block_id * blockDim.x :
                                               blockDim.x;
    }
    else if(m_type == ParallelForType::GridStrideLoop)
    {
        return m_active_num_in_block;
    }
    else
    {
        // Fallback for compilers that reject host-only logging helpers in device context.
        return 0;
    }
}

MUDA_INLINE MUDA_DEVICE bool ParallelForDetails::is_final_block() const MUDA_NOEXCEPT
{
    if(m_type == ParallelForType::DynamicBlocks)
    {
        return (blockIdx.x == gridDim.x - 1);
    }
    else if(m_type == ParallelForType::GridStrideLoop)
    {
        return m_active_num_in_block == blockDim.x;
    }
    else
    {
        // Fallback for compilers that reject host-only logging helpers in device context.
        return false;
    }
}
}  // namespace muda