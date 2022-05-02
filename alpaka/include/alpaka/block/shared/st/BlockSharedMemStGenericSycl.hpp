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

#    include <alpaka/block/shared/st/Traits.hpp>
#    include <alpaka/block/shared/st/detail/BlockSharedMemStMemberImpl.hpp>

#    include <CL/sycl.hpp>

#    include <cstdint>

namespace alpaka::experimental
{
    //! The generic SYCL shared memory allocator.
    class BlockSharedMemStGenericSycl
        : public alpaka::detail::BlockSharedMemStMemberImpl<>
        , public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStGenericSycl>
    {
    public:
        BlockSharedMemStGenericSycl(
            sycl::accessor<std::byte, 1, sycl::access_mode::read_write, sycl::target::local> accessor)
            : BlockSharedMemStMemberImpl(
                reinterpret_cast<std::uint8_t*>(accessor.get_pointer().get()),
                accessor.size())
            , m_accessor{accessor}
        {
        }

    private:
        sycl::accessor<std::byte, 1, sycl::access_mode::read_write, sycl::target::local> m_accessor;
    };
} // namespace alpaka::experimental

namespace alpaka::trait
{
    template<typename T, std::size_t TUniqueId>
    struct DeclareSharedVar<T, TUniqueId, experimental::BlockSharedMemStGenericSycl>
    {
        static auto declareVar(experimental::BlockSharedMemStGenericSycl const& smem) -> T&
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
    struct FreeSharedVars<experimental::BlockSharedMemStGenericSycl>
    {
        static auto freeVars(experimental::BlockSharedMemStGenericSycl const&) -> void
        {
            // shared memory block data will be reused
        }
    };
} // namespace alpaka::trait

#endif
