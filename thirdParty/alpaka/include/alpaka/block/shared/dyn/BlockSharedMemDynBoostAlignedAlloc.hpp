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

#include <alpaka/core/Vectorize.hpp>            // defaultAlignment
#include <alpaka/block/shared/dyn/Traits.hpp>   // AllocVar

#include <alpaka/core/Common.hpp>               // ALPAKA_FN_*

#include <boost/align.hpp>                      // boost::aligned_alloc

#include <vector>                               // std::vector
#include <memory>                               // std::unique_ptr

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
                //#############################################################################
                class BlockSharedMemDynBoostAlignedAlloc
                {
                public:
                    using BlockSharedMemDynBase = BlockSharedMemDynBoostAlignedAlloc;

                    //-----------------------------------------------------------------------------
                    //! Default constructor.
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
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA BlockSharedMemDynBoostAlignedAlloc(BlockSharedMemDynBoostAlignedAlloc const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA BlockSharedMemDynBoostAlignedAlloc(BlockSharedMemDynBoostAlignedAlloc &&) = delete;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSharedMemDynBoostAlignedAlloc const &) -> BlockSharedMemDynBoostAlignedAlloc & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSharedMemDynBoostAlignedAlloc &&) -> BlockSharedMemDynBoostAlignedAlloc & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~BlockSharedMemDynBoostAlignedAlloc() = default;

                public:
                    std::unique_ptr<
                        uint8_t,
                        boost::alignment::aligned_delete> mutable
                            m_blockSharedMemDyn;  //!< Block shared dynamic memory.
                };

                namespace traits
                {
                    //#############################################################################
                    //!
                    //#############################################################################
                    template<
                        typename T>
                    struct GetMem<
                        T,
                        BlockSharedMemDynBoostAlignedAlloc>
                    {
                        //-----------------------------------------------------------------------------
                        //
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
