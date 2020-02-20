/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/block/sync/Traits.hpp>

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <catch2/catch.hpp>

//#############################################################################
class BlockSyncTestKernel
{
public:
    static const std::uint8_t gridThreadExtentPerDim = 4u;

    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success) const
    -> void
    {
        using Idx = alpaka::idx::Idx<TAcc>;

        // Get the index of the current thread within the block and the block extent and map them to 1D.
        auto const blockThreadIdx = alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const blockThreadExtent = alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        auto const blockThreadIdx1D = alpaka::idx::mapIdx<1u>(blockThreadIdx, blockThreadExtent)[0u];
        auto const blockThreadExtent1D = blockThreadExtent.prod();

        // Allocate shared memory.
        Idx * const pBlockSharedArray = alpaka::block::shared::dyn::getMem<Idx>(acc);
   
        // Write the thread index into the shared memory.
        pBlockSharedArray[blockThreadIdx1D] = blockThreadIdx1D;

        // Synchronize the threads in the block.
        alpaka::block::sync::syncBlockThreads(acc);

        // All other threads within the block should now have written their index into the shared memory.
        for(auto i(static_cast<Idx>(0u)); i < blockThreadExtent1D; ++i)
        {
            ALPAKA_CHECK(*success, pBlockSharedArray[i] == i);
        }
    }
};

namespace alpaka
{
    namespace kernel
    {
        namespace traits
        {
            //#############################################################################
            //! The trait for getting the size of the block shared dynamic memory for a kernel.
            template<
                typename TAcc>
            struct BlockSharedMemDynSizeBytes<
                BlockSyncTestKernel,
                TAcc>
            {
                //-----------------------------------------------------------------------------
                //! \return The size of the shared memory allocated for a block.
                template<
                    typename TVec>
                ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
                    BlockSyncTestKernel const & blockSharedMemDyn,
                    TVec const & blockThreadExtent,
                    TVec const & threadElemExtent,
                    bool * success)
                -> idx::Idx<TAcc>
                {
                    using Idx = alpaka::idx::Idx<TAcc>;

                    alpaka::ignore_unused(blockSharedMemDyn);
                    alpaka::ignore_unused(threadElemExtent);
                    alpaka::ignore_unused(success);
                    return
                        static_cast<idx::Idx<TAcc>>(sizeof(Idx)) * blockThreadExtent.prod();
                }
            };
        }
    }
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "synchronize", "[blockSync]", alpaka::test::acc::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::dim::Dim<Acc>;
    using Idx = alpaka::idx::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(
        alpaka::vec::Vec<Dim, Idx>::all(static_cast<Idx>(BlockSyncTestKernel::gridThreadExtentPerDim)));

    BlockSyncTestKernel kernel;

    REQUIRE(
        fixture(
            kernel));
}
