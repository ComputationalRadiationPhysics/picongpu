/* Copyright 2023 Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/block/shared/dyn/Traits.hpp"

#include <cstddef>

#ifdef ALPAKA_ACC_SYCL_ENABLED
#    include <sycl/sycl.hpp>

namespace alpaka
{
    //! The SYCL block shared memory allocator.
    class BlockSharedMemDynGenericSycl
        : public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynGenericSycl>
    {
    public:
        using BlockSharedMemDynBase = BlockSharedMemDynGenericSycl;

        BlockSharedMemDynGenericSycl(sycl::local_accessor<std::byte> accessor) : m_accessor{accessor}
        {
        }

        sycl::local_accessor<std::byte> m_accessor;
    };
} // namespace alpaka

namespace alpaka::trait
{
    template<typename T>
    struct GetDynSharedMem<T, BlockSharedMemDynGenericSycl>
    {
        static auto getMem(BlockSharedMemDynGenericSycl const& shared) -> T*
        {
            return reinterpret_cast<T*>(shared.m_accessor.get_pointer().get());
        }
    };
} // namespace alpaka::trait

#endif
