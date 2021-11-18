/* Copyright 2021 Jan Stephan
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

class DeviceFenceTestKernel
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
            alpaka::mem_fence(acc, alpaka::memory_scope::Device{});
            vars[1] = 20;
        }

        auto const b = vars[1];
        alpaka::mem_fence(acc, alpaka::memory_scope::Device{});
        auto const a = vars[0];

        // If the fence is working correctly, the following case can never happen
        ALPAKA_CHECK(*success, (a != 1 && b == 20));
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
        ALPAKA_CHECK(*success, (a != 1 && b == 20));
    }
};

namespace alpaka
{
    namespace traits
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
    } // namespace traits
} // namespace alpaka

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::size_t>;

TEMPLATE_LIST_TEST_CASE("FenceTest", "[fence]", TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    using Dev = alpaka::Dev<Acc>;
    using Pltf = alpaka::Pltf<Dev>;
    using Queue = alpaka::Queue<Dev, alpaka::property::NonBlocking>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    auto const host = alpaka::getDevByIdx<alpaka::PltfCpu>(0u);
    auto const dev = alpaka::getDevByIdx<Pltf>(0u);
    auto queue = Queue{dev};

    auto const numElements = Idx{2ul};
    auto const extent = alpaka::Vec<Dim, Idx>{numElements};
    auto vars_host = alpaka::allocBuf<int, Idx>(host, extent);
    auto vars_dev = alpaka::allocBuf<int, Idx>(dev, extent);

    auto const pVarsHost = alpaka::getPtrNative(vars_host);
    pVarsHost[0] = 1;
    pVarsHost[1] = 2;

    alpaka::memcpy(queue, vars_dev, vars_host, extent);
    alpaka::wait(queue);

    DeviceFenceTestKernel deviceKernel;
    REQUIRE(fixture(deviceKernel, alpaka::getPtrNative(vars_dev)));

    BlockFenceTestKernel blockKernel;
    REQUIRE(fixture(blockKernel));
}
