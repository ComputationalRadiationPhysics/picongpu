/* Copyright 2023 Jan Stephan, Luca Ferragina, Andrea Bocci, Tapish Narwal
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/fence/Traits.hpp"
#include "alpaka/mem/order/MemoryOrderGenericSycl.hpp"

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka
{
    namespace detail
    {
        template<typename TAlpakaMemScope>
        struct SyclFenceProps
        {
        };

        template<>
        struct SyclFenceProps<alpaka::memory_scope::Block>
        {
            static constexpr auto scope = sycl::memory_scope::work_group;
        };

        template<>
        struct SyclFenceProps<alpaka::memory_scope::Device>
        {
            static constexpr auto scope = sycl::memory_scope::device;
        };

        template<>
        struct SyclFenceProps<alpaka::memory_scope::Grid>
        {
            static constexpr auto scope = sycl::memory_scope::device;
        };
    } // namespace detail

    //! The SYCL memory fence.
    class MemFenceGenericSycl : public interface::Implements<ConceptMemFence, MemFenceGenericSycl>
    {
    };
} // namespace alpaka

namespace alpaka::trait
{
    template<>
    struct MemFenceDefaultOrder<MemFenceGenericSycl>
    {
        using type = mem_order::AcqRel;
        static constexpr auto value = mem_order::acq_rel;
    };

    template<MemoryOrder TMemOrder, MemoryScope TMemScope>
    struct MemFence<MemFenceGenericSycl, TMemOrder, TMemScope>
    {
        static auto mem_fence(MemFenceGenericSycl const&, TMemOrder order, TMemScope const&)
        {
            static constexpr auto scope = alpaka::detail::SyclFenceProps<TMemScope>::scope;
            sycl::atomic_fence(MemOrderSycl::get(order), scope);
        }
    };
} // namespace alpaka::trait

#endif
