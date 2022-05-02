/* Copyright 2022 Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/mem/fence/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch.hpp>

// Trait to detect whether an accelerator supports or not multiple threads per block
template<typename TAcc>
struct IsSingleThreaded : public std::false_type
{
};

/* TODO: Remove the following pragmas once support for clang 5 and 6 is removed. They are necessary because these
/  clang versions incorrectly warn about a missing 'extern'. */
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wmissing-variable-declarations"
#endif
template<typename TAcc>
inline constexpr bool isSingleThreaded = IsSingleThreaded<TAcc>::value;
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
template<typename TDim, typename TIdx>
struct IsSingleThreaded<alpaka::AccCpuSerial<TDim, TIdx>> : public std::true_type
{
};
#endif // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
template<typename TDim, typename TIdx>
struct IsSingleThreaded<alpaka::AccCpuOmp2Blocks<TDim, TIdx>> : public std::true_type
{
};
#endif // ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
template<typename TDim, typename TIdx>
struct IsSingleThreaded<alpaka::AccCpuTbbBlocks<TDim, TIdx>> : public std::true_type
{
};
#endif // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED


class DeviceFenceTestKernelWriter
{
public:
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, ALPAKA_DEVICE_VOLATILE int* vars) const -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];

        // Use a single writer thread
        if(idx == 0)
        {
            vars[0] = 10;
            alpaka::mem_fence(acc, alpaka::memory_scope::Device{});
            vars[1] = 20;
        }
    }
};

class DeviceFenceTestKernelReader
{
public:
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, ALPAKA_DEVICE_VOLATILE int* vars) const -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];

        // Use a single reader thread
        if(idx == 0)
        {
            auto const b = vars[1];
            alpaka::mem_fence(acc, alpaka::memory_scope::Device{});
            auto const a = vars[0];

            // If the fence is working correctly, the following case can never happen
            ALPAKA_CHECK(*success, !(a == 1 && b == 20));
        }
    }
};

class GridFenceTestKernel
{
public:
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success, ALPAKA_DEVICE_VOLATILE int* vars) const -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];

        // Global thread 0 is producer
        if(idx == 0)
        {
            vars[0] = 10;
            alpaka::mem_fence(acc, alpaka::memory_scope::Grid{});
            vars[1] = 20;
        }

        auto const b = vars[1];
        alpaka::mem_fence(acc, alpaka::memory_scope::Grid{});
        auto const a = vars[0];

        // If the fence is working correctly, the following case can never happen
        ALPAKA_CHECK(*success, !(a == 1 && b == 20));
    }
};

class BlockFenceTestKernel
{
public:
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
        auto shared = const_cast<ALPAKA_DEVICE_VOLATILE int*>(alpaka::getDynSharedMem<int>(acc));

        // Initialize
        if(idx == 0)
        {
            shared[0] = 1;
            shared[1] = 2;
        }
        alpaka::syncBlockThreads(acc);

        // Local thread 0 is producer
        if(idx == 0)
        {
            shared[0] = 10;
            alpaka::mem_fence(acc, alpaka::memory_scope::Block{});
            shared[1] = 20;
        }

        auto const b = shared[1];
        alpaka::mem_fence(acc, alpaka::memory_scope::Block{});
        auto const a = shared[0];

        // If the fence is working correctly, the following case can never happen
        ALPAKA_CHECK(*success, !(a == 1 && b == 20));
    }
};

namespace alpaka::trait
{
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template<typename TAcc>
    struct BlockSharedMemDynSizeBytes<BlockFenceTestKernel, TAcc>
    {
        //! \return The size of the shared memory allocated for a block.
        template<typename TVec, typename... TArgs>
        ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            BlockFenceTestKernel const&,
            TVec const&,
            TVec const&,
            TArgs&&...) -> std::size_t
        {
            return 2 * sizeof(int);
        }
    };
} // namespace alpaka::trait

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::size_t>;

TEMPLATE_LIST_TEST_CASE("FenceTest", "[fence]", TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

    using Dev = alpaka::Dev<Acc>;
    using Pltf = alpaka::Pltf<Dev>;
    using Queue = alpaka::Queue<Dev, alpaka::property::NonBlocking>;

    // Fixtures with different number of blocks, threads and elements
    const alpaka::Vec<Dim, Idx> one = {1};
    const alpaka::Vec<Dim, Idx> two = {2};
    alpaka::test::KernelExecutionFixture<Acc> fixtureSingleElement{WorkDiv{one, one, one}};
    alpaka::test::KernelExecutionFixture<Acc> fixtureTwoBlocks{WorkDiv{two, one, one}};
    alpaka::test::KernelExecutionFixture<Acc> fixtureTwoElements = isSingleThreaded<Acc>
        ? alpaka::test::KernelExecutionFixture<Acc>{WorkDiv{one, one, two}}
        : alpaka::test::KernelExecutionFixture<Acc>{WorkDiv{one, two, one}};

    auto const host = alpaka::getDevByIdx<alpaka::PltfCpu>(0u);
    auto const dev = alpaka::getDevByIdx<Pltf>(0u);
    auto queue = Queue{dev};

    auto const numElements = Idx{2ul};
    auto const extent = alpaka::Vec<Dim, Idx>{numElements};
    auto vars_host = alpaka::allocBuf<int, Idx>(host, extent);
    auto vars_dev = alpaka::allocBuf<int, Idx>(dev, extent);
    vars_host[0] = 1;
    vars_host[1] = 2;

    // Run a single kernel, testing a memory fence in shared memory across threads in the same blocks
    BlockFenceTestKernel blockKernel;
    REQUIRE(fixtureTwoElements(blockKernel));

    // Run a single kernel, testing a memory fence in global memory across threads in different blocks
    alpaka::memcpy(queue, vars_dev, vars_host);
    alpaka::wait(queue);
    GridFenceTestKernel gridKernel;
    REQUIRE(fixtureTwoBlocks(gridKernel, vars_dev.data()));

    // Run two kernels in parallel, in two different queues on the same device, testing a memory fence
    // in global memory across threads in different grids
    alpaka::memcpy(queue, vars_dev, vars_host);
    alpaka::wait(queue);
    auto deviceKernelWriter = DeviceFenceTestKernelWriter{};
    auto deviceKernelReader = DeviceFenceTestKernelReader{};
    auto workDiv = WorkDiv{one, one, one};
    alpaka::exec<Acc>(queue, workDiv, deviceKernelWriter, vars_dev.data());
    REQUIRE(fixtureSingleElement(deviceKernelReader, vars_dev.data()));

    alpaka::wait(queue);
}
