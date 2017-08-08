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

#include <alpaka/block/sync/Traits.hpp> // SyncBlockThreads

#include <alpaka/core/Common.hpp>       // ALPAKA_FN_*

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

namespace alpaka
{
    namespace block
    {
        namespace sync
        {
            //#############################################################################
            //! The no op block synchronization.
            //#############################################################################
            class BlockSyncNoOp
            {
            public:
                using BlockSyncBase = BlockSyncNoOp;

                //-----------------------------------------------------------------------------
                //! Default constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC BlockSyncNoOp() = default;
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC BlockSyncNoOp(BlockSyncNoOp const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC BlockSyncNoOp(BlockSyncNoOp &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC auto operator=(BlockSyncNoOp const &) -> BlockSyncNoOp & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC auto operator=(BlockSyncNoOp &&) -> BlockSyncNoOp & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC /*virtual*/ ~BlockSyncNoOp() = default;
            };

            namespace traits
            {
                //#############################################################################
                //!
                //#############################################################################
                template<>
                struct SyncBlockThreads<
                    BlockSyncNoOp>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_ACC static auto syncBlockThreads(
                        block::sync::BlockSyncNoOp const & /*blockSync*/)
                    -> void
                    {
                        //boost::ignore_unused(blockSync);
                        // Nothing to do.
                    }
                };

                //#############################################################################
                //!
                //#############################################################################
                template<
                    typename TOp>
                struct SyncBlockThreadsPredicate<
                    TOp,
                    BlockSyncNoOp>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_ACC static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncNoOp const & /*blockSync*/,
                        int predicate)
                    -> int
                    {
                        //boost::ignore_unused(blockSync);
                        return predicate;
                    }
                };
            }
        }
    }
}
