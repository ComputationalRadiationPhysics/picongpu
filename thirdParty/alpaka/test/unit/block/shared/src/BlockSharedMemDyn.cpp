/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#include <catch2/catch.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>


//#############################################################################
class BlockSharedMemDynTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success) const
    -> void
    {
        // Assure that the pointer is non null.
        auto && a = alpaka::block::shared::dyn::getMem<std::uint32_t>(acc);
        ALPAKA_CHECK(*success, static_cast<std::uint32_t *>(nullptr) != a);

        // Each call should return the same pointer ...
        auto && b = alpaka::block::shared::dyn::getMem<std::uint32_t>(acc);
        ALPAKA_CHECK(*success, a == b);

        // ... even for different types.
        auto && c = alpaka::block::shared::dyn::getMem<float>(acc);
        ALPAKA_CHECK(*success, a == reinterpret_cast<std::uint32_t *>(c));
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
                BlockSharedMemDynTestKernel,
                TAcc>
            {
                //-----------------------------------------------------------------------------
                //! \return The size of the shared memory allocated for a block.
                template<
                    typename TVec>
                ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
                    BlockSharedMemDynTestKernel const & blockSharedMemDyn,
                    TVec const & blockThreadExtent,
                    TVec const & threadElemExtent,
                    bool * success)
                -> idx::Idx<TAcc>
                {
                    alpaka::ignore_unused(blockSharedMemDyn);
                    alpaka::ignore_unused(success);
                    return
                        static_cast<idx::Idx<TAcc>>(sizeof(std::uint32_t)) * blockThreadExtent.prod() * threadElemExtent.prod();
                }
            };
        }
    }
}

//-----------------------------------------------------------------------------
struct TestTemplate
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    BlockSharedMemDynTestKernel kernel;

    REQUIRE(
        fixture(
            kernel));
}
};

TEST_CASE( "sameNonNullAdress", "[blockSharedMemDyn]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplate() );
}
