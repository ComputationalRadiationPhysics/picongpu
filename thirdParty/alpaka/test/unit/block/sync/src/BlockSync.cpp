/**
 * \file
 * Copyright 2017 Benjamin Worpitz
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

// \Hack: Boost.MPL defines BOOST_MPL_CFG_GPU_ENABLED to __host__ __device__ if nvcc is used.
// BOOST_AUTO_TEST_CASE_TEMPLATE and its internals are not GPU enabled but is using boost::mpl::for_each internally.
// For each template parameter this leads to:
// /home/travis/build/boost/boost/mpl/for_each.hpp(78): warning: calling a __host__ function from a __host__ __device__ function is not allowed
// because boost::mpl::for_each has the BOOST_MPL_CFG_GPU_ENABLED attribute but the test internals are pure host methods.
// Because we do not use MPL within GPU code here, we can disable the MPL GPU support.
#define BOOST_MPL_CFG_GPU_ENABLED

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <boost/assert.hpp>
#include <boost/predef.h>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

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
        TAcc const & acc) const
    -> void
    {
        using Size = alpaka::size::Size<TAcc>;

        // Get the index of the current thread within the block and the block extent and map them to 1D.
        auto const blockThreadIdx = alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const blockThreadExtent = alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        auto const blockThreadIdx1D = alpaka::idx::mapIdx<1u>(blockThreadIdx, blockThreadExtent)[0u];
        auto const blockThreadExtent1D = blockThreadExtent.prod();

        // Allocate shared memory.
        Size * const pBlockSharedArray = alpaka::block::shared::dyn::getMem<Size>(acc);
   
        // Write the thread index into the shared memory.
        pBlockSharedArray[blockThreadIdx1D] = blockThreadIdx1D;

        // Synchronize the threads in the block.
        alpaka::block::sync::syncBlockThreads(acc);

        // All other threads within the block should now have written their index into the shared memory.
        for(auto i(static_cast<Size>(0u)); i < blockThreadExtent1D; ++i)
        {
            BOOST_VERIFY(pBlockSharedArray[i] == i);
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
                ALPAKA_FN_HOST static auto getBlockSharedMemDynSizeBytes(
                    BlockSyncTestKernel const & blockSharedMemDyn,
                    TVec const & blockThreadExtent,
                    TVec const & threadElemExtent)
                -> size::Size<TAcc>
                {
                    using Size = alpaka::size::Size<TAcc>;

                    boost::ignore_unused(blockSharedMemDyn);
                    boost::ignore_unused(threadElemExtent);
                    return
                        static_cast<size::Size<TAcc>>(sizeof(Size)) * blockThreadExtent.prod();
                }
            };
        }
    }
}

BOOST_AUTO_TEST_SUITE(blockSync)

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    synchronize,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Size>::all(static_cast<Size>(BlockSyncTestKernel::gridThreadExtentPerDim)));

    BlockSyncTestKernel kernel;

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
}

BOOST_AUTO_TEST_SUITE_END()
