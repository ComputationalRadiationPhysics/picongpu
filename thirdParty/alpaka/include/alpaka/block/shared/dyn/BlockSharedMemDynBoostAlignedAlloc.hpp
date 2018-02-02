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
                    ALPAKA_FN_ACC_NO_CUDA BlockSharedMemDynBoostAlignedAlloc(
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
                    ALPAKA_FN_ACC_NO_CUDA BlockSharedMemDynBoostAlignedAlloc(BlockSharedMemDynBoostAlignedAlloc const &) = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA BlockSharedMemDynBoostAlignedAlloc(BlockSharedMemDynBoostAlignedAlloc &&) = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSharedMemDynBoostAlignedAlloc const &) -> BlockSharedMemDynBoostAlignedAlloc & = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSharedMemDynBoostAlignedAlloc &&) -> BlockSharedMemDynBoostAlignedAlloc & = delete;
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
                    //#############################################################################
                    template<
                        typename T>
                    struct GetMem<
                        T,
                        BlockSharedMemDynBoostAlignedAlloc>
                    {
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_ACC_NO_CUDA static auto getMem(
                            block::shared::dyn::BlockSharedMemDynBoostAlignedAlloc const & blockSharedMemDyn)
                        -> T *
                        {
                            return reinterpret_cast<T*>(blockSharedMemDyn.m_blockSharedMemDyn.get());
                        }
                    };
                }
            }
        }
    }
}
