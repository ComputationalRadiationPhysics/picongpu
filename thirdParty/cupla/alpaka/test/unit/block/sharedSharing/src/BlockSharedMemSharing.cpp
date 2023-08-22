/* Copyright 2022 Jeffrey Kelling, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/acc/Traits.hpp>
#include <alpaka/atomic/Traits.hpp>
#include <alpaka/block/shared/dyn/Traits.hpp>
#include <alpaka/block/shared/st/Traits.hpp>
#include <alpaka/block/sync/Traits.hpp>
#include <alpaka/test/Array.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/workdiv/Traits.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

//! This tests checks if block-shared memory is shared correctly: only between all threads in a block.
//!
//! The check is done by each thread atomically adding 1 to a variable `shared`. Possible outcomes are:
//! * `shared < blockThreadCount`: memory is not shared between all threads of a block; usually `shared == 1`: memory
//!   is thread-private (fail)
//! * `shared > blockThreadCount`: memory is shared between blocks (fail)
//! * `shared == blockThreadCount`: memory is shared correctly (pass)
template<typename TAcc>
ALPAKA_FN_ACC void blockSharedMemSharingTestKernelHelper(TAcc const& acc, std::uint32_t* sums, std::uint32_t& shared)
{
    auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
    auto const gridBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];

    if(blockThreadIdx == 0u)
        shared = 0u;

    alpaka::syncBlockThreads(acc);

    // In the expected case we only need `hierarchy::Blocks` here, but in
    // case shared memory is shared between blocks we need atomicity
    // between blocks to get the correct result telling us so.
    alpaka::atomicAdd(acc, &shared, 1u, alpaka::hierarchy::Blocks());

    alpaka::syncBlockThreads(acc);

    if(blockThreadIdx == 0u)
        sums[gridBlockIdx] = shared;
}

template<class TAcc, class TKernel>
void BlockSharedMemSharingTest(TKernel kernel)
{
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;
    using Vec = alpaka::Vec<Dim, Idx>;

    auto const platformAcc = alpaka::Platform<TAcc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    auto const accDevProps = alpaka::getAccDevProps<TAcc>(devAcc);
    const Idx gridBlockCount = 2u;
    const Idx blockThreadCount = accDevProps.m_blockThreadCountMax;

    auto const workDiv
        = alpaka::WorkDivMembers<Dim, Idx>(Vec(gridBlockCount), Vec(blockThreadCount), Vec(static_cast<Idx>(1u)));

    auto queue = alpaka::Queue<TAcc, alpaka::Blocking>(devAcc);

    auto bufAcc = alpaka::allocBuf<std::uint32_t, Idx>(devAcc, gridBlockCount);

    alpaka::exec<TAcc>(queue, workDiv, kernel, alpaka::getPtrNative(bufAcc));

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    auto bufHost = alpaka::allocBuf<std::uint32_t, Idx>(devHost, gridBlockCount);

    alpaka::memcpy(queue, bufHost, bufAcc);

    auto pBufHost = alpaka::getPtrNative(bufHost);
    for(Idx a = 0u; a < gridBlockCount; ++a)
    {
        REQUIRE(pBufHost[a] == blockThreadCount);
    }
}

class BlockSharedMemStSharingTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, std::uint32_t* sums) const
    {
        auto& shared = alpaka::declareSharedVar<std::uint32_t, __COUNTER__>(acc);

        blockSharedMemSharingTestKernelHelper(acc, sums, shared);
    }
};

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::uint32_t>;

TEMPLATE_LIST_TEST_CASE("blockSharedMemSt", "[blockSharedMemSharing]", TestAccs)
{
    using Acc = TestType;

    BlockSharedMemStSharingTestKernel kernel;

    BlockSharedMemSharingTest<Acc>(kernel);
}

class BlockSharedMemDynSharingTestKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, std::uint32_t* sums) const
    {
        auto& shared = *alpaka::getDynSharedMem<std::uint32_t>(acc);

        blockSharedMemSharingTestKernelHelper(acc, sums, shared);
    }
};

namespace alpaka::trait
{
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template<typename TAcc>
    struct BlockSharedMemDynSizeBytes<BlockSharedMemDynSharingTestKernel, TAcc>
    {
        //! \return The size of the shared memory allocated for a block.
        template<typename TVec>
        ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            BlockSharedMemDynSharingTestKernel const& /* blockSharedMemDyn */,
            TVec const& /* blockThreadExtent */,
            TVec const& /* threadElemExtent */,
            std::uint32_t* /* sums */) -> std::size_t
        {
            return sizeof(std::uint32_t);
        }
    };
} // namespace alpaka::trait

TEMPLATE_LIST_TEST_CASE("blockSharedMemDyn", "[blockSharedMemSharing]", TestAccs)
{
    using Acc = TestType;

    BlockSharedMemDynSharingTestKernel kernel;

    BlockSharedMemSharingTest<Acc>(kernel);
}
