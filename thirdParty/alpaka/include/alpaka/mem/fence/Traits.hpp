/* Copyright 2022 Jan Stephan, Andrea Bocci, Tapish Narwal
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Interface.hpp"
#include "alpaka/mem/order/MemoryOrder.hpp"

namespace alpaka
{
    struct ConceptMemFence
    {
    };

    namespace memory_scope
    {
        struct MemoryScopeTag
        {
        };

        //! Memory fences are observed by all threads in the same block.
        struct Block : MemoryScopeTag
        {
        };

        //! Memory fences are observed by all threads in the same grid.
        struct Grid : MemoryScopeTag
        {
        };

        //! Memory fences are observed by all threads on the device.
        struct Device : MemoryScopeTag
        {
        };
    } // namespace memory_scope

    template<typename T>
    concept MemoryScope = std::derived_from<T, memory_scope::MemoryScopeTag>;

    //! The memory fence trait.
    namespace trait
    {
        //! The mem_fence trait.
        template<typename TMemFence, MemoryOrder TMemOrder, MemoryScope TMemScope, typename TSfinae = void>
        struct MemFence;

        template<typename TAcc>
        struct MemFenceDefaultOrder;

        template<typename TAcc>
        using MemFenceDefaultOrder_t = typename MemFenceDefaultOrder<TAcc>::type;

        template<typename TAcc>
        inline constexpr auto MemFenceDefaultOrder_v = MemFenceDefaultOrder<TAcc>::value;

    } // namespace trait

    //! Issues memory fence instructions.
    //
    // Issues a memory fence instruction for a given memory scope (\a memory_scope::Block or \a memory_scope::Device).
    // This guarantees the following:
    // * All \a LOAD instructions preceeding the fence will always occur before the LOAD instructions following the
    //   fence (\a LoadLoad coherence)
    // * All \a STORE instructions preceeding the fence will always occur before the STORE instructions following the
    //   fence (\a LoadStore coherence). The pre-fence STORE results will be propagated to the other threads in the
    //   scope at an unknown point in time.
    //
    // Note that there are no further guarantees, especially with regard to \a LoadStore ordering. Users should not
    // mistake this as a synchronization function between threads (please use syncBlockThreads() instead).
    //
    //! \tparam TMemFence The memory fence implementation type.
    //! \tparam TMemScope The memory scope type.
    //! \tparam TMemOrder The memory order type.
    //! \param fence The memory fence implementation.
    //! \param scope The memory scope.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TMemFence, MemoryOrder TMemOrder, MemoryScope TMemScope>
    ALPAKA_FN_ACC auto mem_fence(TMemFence const& fence, TMemOrder order, TMemScope const& scope) -> void
    {
        using ImplementationBase = interface::ImplementationBase<ConceptMemFence, TMemFence>;
        if constexpr(std::is_same_v<TMemOrder, mem_order::Relaxed>)
        {
            // Relaxed ordering requires no fence.
            // Relaxed memory fences make no sense at all anyway. It is an oxymoron. This should not be used.
            // STL says it is a noop. https://en.cppreference.com/w/cpp/atomic/atomic_thread_fence.html
            // OpenMP does not provide a relaxed flush at all. https://www.openmp.org/spec-html/5.0/openmpsu96.html
            // Sycl says it is a noop. https://github.khronos.org/SYCL_Reference/iface/barriers-and-fences.html
            // When using relaxed with mem fences, nvcc generates PTX for a sequenitally consistent fence
            // This may be a problem also with HIP, so we explicitly skip it for all backends
        }
        else
        {
            trait::MemFence<ImplementationBase, TMemOrder, TMemScope>::mem_fence(fence, order, scope);
        }
    }

    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TMemFence, MemoryScope TMemScope>
    ALPAKA_FN_ACC auto mem_fence(TMemFence const& fence, TMemScope const& scope) -> void
    {
        using ImplementationBase = interface::ImplementationBase<ConceptMemFence, TMemFence>;
        mem_fence(fence, trait::MemFenceDefaultOrder_v<ImplementationBase>, scope);
    }

} // namespace alpaka
