/* Copyright 2019 Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Vectorize.hpp>
#include <alpaka/block/shared/dyn/Traits.hpp>

#include <alpaka/core/Common.hpp>

#include <boost/align.hpp>

#include <vector>
#include <memory>

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            namespace dyn
            {
                //#############################################################################
                //! The block shared dynamic memory allocator without synchronization.
                class BlockSharedMemDynBoostAlignedAlloc : public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynBoostAlignedAlloc>
                {
                public:
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynBoostAlignedAlloc(
                        std::size_t const & blockSharedMemDynSizeBytes)
                    {
                        if(blockSharedMemDynSizeBytes > 0u)
                        {
                            m_blockSharedMemDyn.reset(
                                reinterpret_cast<uint8_t *>(
                                    boost::alignment::aligned_alloc(core::vectorization::defaultAlignment, blockSharedMemDynSizeBytes)));
                        }
                    }
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynBoostAlignedAlloc(BlockSharedMemDynBoostAlignedAlloc const &) = delete;
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynBoostAlignedAlloc(BlockSharedMemDynBoostAlignedAlloc &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemDynBoostAlignedAlloc const &) -> BlockSharedMemDynBoostAlignedAlloc & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemDynBoostAlignedAlloc &&) -> BlockSharedMemDynBoostAlignedAlloc & = delete;
                    //-----------------------------------------------------------------------------
                    /*virtual*/ ~BlockSharedMemDynBoostAlignedAlloc() = default;

                public:
                    std::unique_ptr<
                        uint8_t,
                        boost::alignment::aligned_delete> mutable
                            m_blockSharedMemDyn;  //!< Block shared dynamic memory.
                };

                namespace traits
                {
#if BOOST_COMP_GNUC
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wcast-align" // "cast from 'unsigned char*' to 'unsigned int*' increases required alignment of target type"
#endif
                    //#############################################################################
                    template<
                        typename T>
                    struct GetMem<
                        T,
                        BlockSharedMemDynBoostAlignedAlloc>
                    {
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST static auto getMem(
                            block::shared::dyn::BlockSharedMemDynBoostAlignedAlloc const & blockSharedMemDyn)
                        -> T *
                        {
                            static_assert(
                                core::vectorization::defaultAlignment >= alignof(T),
                                "Unable to get block shared dynamic memory for types with alignment higher than defaultAlignment!");

                            return reinterpret_cast<T*>(blockSharedMemDyn.m_blockSharedMemDyn.get());
                        }
                    };
#if BOOST_COMP_GNUC
    #pragma GCC diagnostic pop
#endif
                }
            }
        }
    }
}
