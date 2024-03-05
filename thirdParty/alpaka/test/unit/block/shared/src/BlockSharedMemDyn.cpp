/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/block/shared/dyn/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

class BlockSharedMemDynTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        // Assure that the pointer is non null.
        auto a = alpaka::getDynSharedMem<std::uint32_t>(acc);
        ALPAKA_CHECK(*success, static_cast<std::uint32_t*>(nullptr) != a);

        // Each call should return the same pointer ...
        auto b = alpaka::getDynSharedMem<std::uint32_t>(acc);
        ALPAKA_CHECK(*success, a == b);

        // ... even for different types.
        auto c = alpaka::getDynSharedMem<float>(acc);
        ALPAKA_CHECK(*success, a == reinterpret_cast<std::uint32_t*>(c));
    }
};

namespace alpaka::trait
{
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template<typename TAcc>
    struct BlockSharedMemDynSizeBytes<BlockSharedMemDynTestKernel, TAcc>
    {
        //! \return The size of the shared memory allocated for a block.
        template<typename TVec>
        ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            BlockSharedMemDynTestKernel const& /* blockSharedMemDyn */,
            TVec const& blockThreadExtent,
            TVec const& threadElemExtent,
            bool* /* success */) -> std::size_t
        {
            auto const gridSize = blockThreadExtent.prod() * threadElemExtent.prod();
            return static_cast<std::size_t>(gridSize) * sizeof(std::uint32_t);
        }
    };
} // namespace alpaka::trait

TEMPLATE_LIST_TEST_CASE("sameNonNullAdress", "[blockSharedMemDyn]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    BlockSharedMemDynTestKernel kernel;

    REQUIRE(fixture(kernel));
}
