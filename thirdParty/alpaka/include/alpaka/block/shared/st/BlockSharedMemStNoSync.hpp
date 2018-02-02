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
#include <alpaka/block/shared/st/Traits.hpp>

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
            namespace st
            {
                //#############################################################################
                //! The block shared memory allocator without synchronization.
                class BlockSharedMemStNoSync
                {
                public:
                    using BlockSharedMemStBase = BlockSharedMemStNoSync;

                    //-----------------------------------------------------------------------------
                    BlockSharedMemStNoSync() = default;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA BlockSharedMemStNoSync(BlockSharedMemStNoSync const &) = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA BlockSharedMemStNoSync(BlockSharedMemStNoSync &&) = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSharedMemStNoSync const &) -> BlockSharedMemStNoSync & = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSharedMemStNoSync &&) -> BlockSharedMemStNoSync & = delete;
                    //-----------------------------------------------------------------------------
                    /*virtual*/ ~BlockSharedMemStNoSync() = default;

                public:
                    // TODO: We should add the size of the (current) allocation.
                    // This would allow to assert that all parallel function calls request to allocate the same size.
                    std::vector<
                        std::unique_ptr<
                            uint8_t,
                            boost::alignment::aligned_delete>> mutable
                        m_sharedAllocs;
                };

                namespace traits
                {
                    //#############################################################################
                    template<
                        typename T,
                        std::size_t TuniqueId>
                    struct AllocVar<
                        T,
                        TuniqueId,
                        BlockSharedMemStNoSync>
                    {
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_ACC_NO_CUDA static auto allocVar(
                            block::shared::st::BlockSharedMemStNoSync const & blockSharedMemSt)
                        -> T &
                        {
                            blockSharedMemSt.m_sharedAllocs.emplace_back(
                                reinterpret_cast<uint8_t *>(
                                    boost::alignment::aligned_alloc(core::vectorization::defaultAlignment, sizeof(T))));
                            return
                                std::ref(
                                    *reinterpret_cast<T*>(
                                        blockSharedMemSt.m_sharedAllocs.back().get()));
                        }
                    };
                    //#############################################################################
                    template<>
                    struct FreeMem<
                        BlockSharedMemStNoSync>
                    {
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_ACC_NO_CUDA static auto freeMem(
                            block::shared::st::BlockSharedMemStNoSync const & blockSharedMemSt)
                        -> void
                        {
                            blockSharedMemSt.m_sharedAllocs.clear();
                        }
                    };
                }
            }
        }
    }
}
