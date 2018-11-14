/**
* \file
* Copyright 2014-2015 Benjamin Worpitz, Rene Widera
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
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
                class BlockSharedMemDynBoostAlignedAlloc
                {
                public:
                    using BlockSharedMemDynBase = BlockSharedMemDynBoostAlignedAlloc;

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
