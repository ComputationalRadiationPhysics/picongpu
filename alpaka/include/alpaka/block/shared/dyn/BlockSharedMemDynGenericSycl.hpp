/* Copyright 2022 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/block/shared/dyn/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <cstddef>

namespace alpaka::experimental
{
    //! The SYCL block shared memory allocator.
    class BlockSharedMemDynGenericSycl
        : public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynGenericSycl>
    {
    public:
        using BlockSharedMemDynBase = BlockSharedMemDynGenericSycl;

        BlockSharedMemDynGenericSycl(
            sycl::accessor<std::byte, 1, sycl::access::mode::read_write, sycl::access::target::local> accessor)
            : m_accessor{accessor}
        {
        }

        sycl::accessor<std::byte, 1, sycl::access::mode::read_write, sycl::access::target::local> m_accessor;
    };
} // namespace alpaka::experimental

namespace alpaka::trait
{
    template<typename T>
    struct GetDynSharedMem<T, experimental::BlockSharedMemDynGenericSycl>
    {
        static auto getMem(experimental::BlockSharedMemDynGenericSycl const& shared) -> T*
        {
            auto void_ptr = sycl::multi_ptr<void, sycl::access::address_space::local_space>{shared.m_accessor};
            auto t_ptr = static_cast<sycl::multi_ptr<T, sycl::access::address_space::local_space>>(void_ptr);
            return t_ptr.get();
        }
    };
} // namespace alpaka::trait

#endif
