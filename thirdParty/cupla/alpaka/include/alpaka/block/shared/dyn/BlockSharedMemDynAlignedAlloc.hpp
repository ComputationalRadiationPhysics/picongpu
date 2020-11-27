/* Copyright 2019 Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/block/shared/dyn/Traits.hpp>
#include <alpaka/core/AlignedAlloc.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/Vectorize.hpp>

#include <memory>
#include <vector>

namespace alpaka
{
    //#############################################################################
    //! The block shared dynamic memory allocator without synchronization.
    class BlockSharedMemDynAlignedAlloc
        : public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynAlignedAlloc>
    {
    public:
        //-----------------------------------------------------------------------------
        BlockSharedMemDynAlignedAlloc(std::size_t const& blockSharedMemDynSizeBytes)
        {
            if(blockSharedMemDynSizeBytes > 0u)
            {
                m_blockSharedMemDyn.reset(reinterpret_cast<uint8_t*>(
                    core::alignedAlloc(core::vectorization::defaultAlignment, blockSharedMemDynSizeBytes)));
            }
        }
        //-----------------------------------------------------------------------------
        BlockSharedMemDynAlignedAlloc(BlockSharedMemDynAlignedAlloc const&) = delete;
        //-----------------------------------------------------------------------------
        BlockSharedMemDynAlignedAlloc(BlockSharedMemDynAlignedAlloc&&) = delete;
        //-----------------------------------------------------------------------------
        auto operator=(BlockSharedMemDynAlignedAlloc const&) -> BlockSharedMemDynAlignedAlloc& = delete;
        //-----------------------------------------------------------------------------
        auto operator=(BlockSharedMemDynAlignedAlloc&&) -> BlockSharedMemDynAlignedAlloc& = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~BlockSharedMemDynAlignedAlloc() = default;

    public:
        std::unique_ptr<uint8_t,
                        core::AlignedDelete> mutable m_blockSharedMemDyn; //!< Block shared dynamic memory.
    };

    namespace traits
    {
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored                                                                                    \
        "-Wcast-align" // "cast from 'unsigned char*' to 'unsigned int*' increases required alignment of target type"
#endif
        //#############################################################################
        template<typename T>
        struct GetDynSharedMem<T, BlockSharedMemDynAlignedAlloc>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getMem(BlockSharedMemDynAlignedAlloc const& blockSharedMemDyn) -> T*
            {
                static_assert(
                    core::vectorization::defaultAlignment >= alignof(T),
                    "Unable to get block shared dynamic memory for types with alignment higher than "
                    "defaultAlignment!");

                return reinterpret_cast<T*>(blockSharedMemDyn.m_blockSharedMemDyn.get());
            }
        };
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
    } // namespace traits
} // namespace alpaka
