/* Copyright 2022 Benjamin Worpitz, Erik Zenker, René Widera, Bernhard Manfred Gruber
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#    if _OPENMP < 201307
#        error If ALPAKA_ACC_ANY_BT_OMP5_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#    endif

#    include <alpaka/block/shared/st/Traits.hpp>
#    include <alpaka/block/shared/st/detail/BlockSharedMemStMemberImpl.hpp>

#    include <omp.h>

#    include <cstdint>
#    include <type_traits>

namespace alpaka
{
    //! The OpenMP 5 block shared memory allocator.
    class BlockSharedMemStOmp5
        : public detail::BlockSharedMemStMemberImpl<4>
        , public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStOmp5>
    {
    public:
        using BlockSharedMemStMemberImpl<4>::BlockSharedMemStMemberImpl;
    };

    namespace trait
    {
        template<typename T, std::size_t TuniqueId>
        struct DeclareSharedVar<T, TuniqueId, BlockSharedMemStOmp5>
        {
            static auto declareVar(BlockSharedMemStOmp5 const& smem) -> T&
            {
                auto* data = smem.template getVarPtr<T>(TuniqueId);

                if(!data)
                {
#    pragma omp barrier
#    pragma omp single
                    {
                        smem.template alloc<T>(TuniqueId);
                    }
                    // lookup for the data chunk allocated by the master thread
                    data = smem.template getLatestVarPtr<T>();
                }
                ALPAKA_ASSERT_OFFLOAD(data != nullptr);

                return *data;
            }
        };
        template<>
        struct FreeSharedVars<BlockSharedMemStOmp5>
        {
            static auto freeVars(BlockSharedMemStOmp5 const&) -> void
            {
                // shared memory block data will be reused
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
