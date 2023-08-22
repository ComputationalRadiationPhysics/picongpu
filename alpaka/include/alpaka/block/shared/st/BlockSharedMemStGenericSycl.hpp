/* Copyright 2023 Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/block/shared/st/Traits.hpp"
#include "alpaka/block/shared/st/detail/BlockSharedMemStMemberImpl.hpp"

#include <cstddef>
#include <cstdint>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka
{
    //! The generic SYCL shared memory allocator.
    class BlockSharedMemStGenericSycl
        : public alpaka::detail::BlockSharedMemStMemberImpl<>
        , public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStGenericSycl>
    {
    public:
        BlockSharedMemStGenericSycl(sycl::local_accessor<std::byte> accessor)
            : BlockSharedMemStMemberImpl(
                reinterpret_cast<std::uint8_t*>(accessor.get_pointer().get()),
                accessor.size())
            , m_accessor{accessor}
        {
        }

    private:
        sycl::local_accessor<std::byte> m_accessor;
    };
} // namespace alpaka

namespace alpaka::trait
{
    template<typename T, std::size_t TUniqueId>
    struct DeclareSharedVar<T, TUniqueId, BlockSharedMemStGenericSycl>
    {
        static auto declareVar(BlockSharedMemStGenericSycl const& smem) -> T&
        {
            auto* data = smem.template getVarPtr<T>(TUniqueId);

            if(!data)
            {
                smem.template alloc<T>(TUniqueId);
                data = smem.template getLatestVarPtr<T>();
            }
            ALPAKA_ASSERT(data != nullptr);
            return *data;
        }
    };

    template<>
    struct FreeSharedVars<BlockSharedMemStGenericSycl>
    {
        static auto freeVars(BlockSharedMemStGenericSycl const&) -> void
        {
            // shared memory block data will be reused
        }
    };
} // namespace alpaka::trait

#endif
