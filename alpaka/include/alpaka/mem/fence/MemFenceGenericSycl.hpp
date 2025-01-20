/* Copyright 2023 Jan Stephan, Luca Ferragina, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/fence/Traits.hpp"

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
    class MemFenceGenericSycl : public concepts::Implements<ConceptMemFence, MemFenceGenericSycl>
    {
    };
} // namespace alpaka

namespace alpaka::trait
{
    template<typename TMemScope>
    struct MemFence<MemFenceGenericSycl, TMemScope>
    {
        static auto mem_fence(MemFenceGenericSycl const&, TMemScope const&)
        {
            static constexpr auto scope = alpaka::detail::SyclFenceProps<TMemScope>::scope;
            sycl::atomic_fence(sycl::memory_order::acq_rel, scope);
        }
    };
} // namespace alpaka::trait

#endif
