/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/block/sync/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

class BlockSyncTestKernel
{
public:
    static constexpr std::uint8_t gridThreadExtentPerDim()
    {
        return 4u;
    }

    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        using Idx = alpaka::Idx<TAcc>;

        // Get the index of the current thread within the block and the block extent and map them to 1D.
        auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        auto const blockThreadIdx1D = alpaka::mapIdx<1u>(blockThreadIdx, blockThreadExtent)[0u];
        auto const blockThreadExtent1D = blockThreadExtent.prod();

        // Allocate shared memory.
        Idx* const pBlockSharedArray = alpaka::getDynSharedMem<Idx>(acc);

        // Write the thread index into the shared memory.
        pBlockSharedArray[blockThreadIdx1D] = blockThreadIdx1D;

        // Synchronize the threads in the block.
        alpaka::syncBlockThreads(acc);

        // All other threads within the block should now have written their index into the shared memory.
        for(auto i = static_cast<Idx>(0u); i < blockThreadExtent1D; ++i)
        {
            ALPAKA_CHECK(*success, pBlockSharedArray[i] == i);
        }
    }
};

namespace alpaka::trait
{
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template<typename TAcc>
    struct BlockSharedMemDynSizeBytes<BlockSyncTestKernel, TAcc>
    {
        //! \return The size of the shared memory allocated for a block.
        template<typename TVec>
        ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            BlockSyncTestKernel const& /* blockSharedMemDyn */,
            TVec const& blockThreadExtent,
            TVec const& /* threadElemExtent */,
            bool* /* success */) -> std::size_t
        {
            using Idx = alpaka::Idx<TAcc>;
            return static_cast<std::size_t>(blockThreadExtent.prod()) * sizeof(Idx);
        }
    };
} // namespace alpaka::trait

TEMPLATE_LIST_TEST_CASE("synchronize", "[blockSync]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(
        alpaka::Vec<Dim, Idx>::all(static_cast<Idx>(BlockSyncTestKernel::gridThreadExtentPerDim())));

    BlockSyncTestKernel kernel;

    REQUIRE(fixture(kernel));
}
