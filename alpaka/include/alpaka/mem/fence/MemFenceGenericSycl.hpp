/* Copyright 2022 Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/mem/fence/Traits.hpp>

#    include <CL/sycl.hpp>

namespace alpaka::experimental
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
            static constexpr auto space = sycl::access::address_space::local_space;
        };

        template<>
        struct SyclFenceProps<alpaka::memory_scope::Device>
        {
            static constexpr auto scope = sycl::memory_scope::device;
            static constexpr auto space = sycl::access::address_space::global_space;
        };
    } // namespace detail

    //! The SYCL memory fence.
    class MemFenceGenericSycl : public concepts::Implements<ConceptMemFence, MemFenceGenericSycl>
    {
    public:
        MemFenceGenericSycl(
            sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::global_buffer> global_dummy,
            sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local> local_dummy)
            : m_global_dummy{global_dummy}
            , m_local_dummy{local_dummy}
        {
        }

        sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::global_buffer> m_global_dummy;
        sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local> m_local_dummy;
    };
} // namespace alpaka::experimental

namespace alpaka::trait
{
    template<typename TMemScope>
    struct MemFence<experimental::MemFenceGenericSycl, TMemScope>
    {
        static auto mem_fence(experimental::MemFenceGenericSycl const& fence, TMemScope const&)
        {
            static constexpr auto scope = experimental::detail::SyclFenceProps<TMemScope>::scope;
            static constexpr auto space = experimental::detail::SyclFenceProps<TMemScope>::space;

            // atomic_ref is already part of the SYCL spec but oneAPI has not caught up yet.
            auto dummy
                = (scope == sycl::memory_scope::work_group)
                      ? sycl::ext::oneapi::
                          atomic_ref<int, sycl::ext::oneapi::memory_order::relaxed, scope, space>{fence.m_local_dummy
                                                                                                      [0]}
                      : sycl::ext::oneapi::atomic_ref<int, sycl::ext::oneapi::memory_order::relaxed, scope, space>{
                          fence.m_global_dummy[0]};
            auto const dummy_val = dummy.load();
            sycl::atomic_fence(sycl::memory_order::acq_rel, scope);
            dummy.store(dummy_val);
        }
    };
} // namespace alpaka::trait

#endif
