/* Copyright 2024 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include "alpaka/exec/Once.hpp"

#include "alpaka/atomic/Traits.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/meta/ForEachType.hpp"
#include "alpaka/test/KernelExecutionFixture.hpp"
#include "alpaka/test/acc/TestAccs.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

class KernelOncePerGrid
{
public:
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* status, int32_t* value) const -> void
    {
        // Only one thread in the whole grid should increment the conter.
        if(alpaka::oncePerGrid(acc))
        {
            ALPAKA_CHECK(*status, *value == 0);

            alpaka::atomicAdd(acc, value, 1, alpaka::hierarchy::Grids{});

            ALPAKA_CHECK(*status, *value == 1);
        }
    }
};

// MSVC does not seem to recognize as "true" a value set to "true" in device code,
// so force all object representations different from zero to evaluate as "true".
inline void fixBooleanValue(bool& value)
{
    value = reinterpret_cast<char const&>(value) == 0x00 ? false : true;
}

TEMPLATE_LIST_TEST_CASE("oncePerGrid", "[exec]", alpaka::test::TestAccs)
{
    using Host = alpaka::DevCpu;
    Host host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);

    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    using Platform = alpaka::Platform<Acc>;
    using Device = alpaka::Dev<Platform>;
    using Queue = alpaka::Queue<Device, alpaka::NonBlocking>;

    Platform platform;
    Device device = alpaka::getDevByIdx(platform, 0);
    Queue queue{device};

    using Scalar = alpaka::Vec<alpaka::DimInt<0u>, Idx>;
    auto value = alpaka::allocMappedBuf<int32_t, Idx>(host, platform, Scalar{});
    *value = 0;

    auto status = alpaka::allocMappedBuf<bool, Idx>(host, platform, Scalar{});
    *status = true;

    auto const extent = alpaka::Vec<Dim, Idx>::all(32);
    auto const elems = alpaka::Vec<Dim, Idx>::all(4);

    KernelOncePerGrid kernel;
    alpaka::KernelCfg<Acc> const config = {extent, elems, false};
    auto const workDiv = alpaka::getValidWorkDiv(config, device, kernel, std::data(status), std::data(value));

    alpaka::exec<Acc>(queue, workDiv, kernel, std::data(status), std::data(value));
    alpaka::wait(queue);

    fixBooleanValue(*status);
    REQUIRE(*status == true);
    REQUIRE(*value == 1);
}

class KernelOncePerBlock
{
public:
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* status, int32_t* value) const -> void
    {
        const int32_t blocks = static_cast<int32_t>(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc).prod());

        // Only one thread in each block should increment the conter.
        if(alpaka::oncePerBlock(acc))
        {
            // FIXME: implement alpaka::atomicLoad and use it here
            int before = alpaka::atomicAdd(acc, value, 0, alpaka::hierarchy::Grids{});
            ALPAKA_CHECK(*status, before >= 0);
            ALPAKA_CHECK(*status, before < blocks);

            alpaka::atomicAdd(acc, value, 1, alpaka::hierarchy::Grids{});

            // FIXME: implement alpaka::atomicLoad and use it here
            int after = alpaka::atomicAdd(acc, value, 0, alpaka::hierarchy::Grids{});
            ALPAKA_CHECK(*status, after > 0);
            ALPAKA_CHECK(*status, after <= blocks);
        }
    }
};

TEMPLATE_LIST_TEST_CASE("oncePerBlock", "[exec]", alpaka::test::TestAccs)
{
    using Host = alpaka::DevCpu;
    Host host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);

    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    using Platform = alpaka::Platform<Acc>;
    using Device = alpaka::Dev<Platform>;
    using Queue = alpaka::Queue<Device, alpaka::NonBlocking>;

    Platform platform;
    Device device = alpaka::getDevByIdx(platform, 0);
    Queue queue{device};

    using Scalar = alpaka::Vec<alpaka::DimInt<0u>, Idx>;
    auto value = alpaka::allocMappedBuf<int32_t, Idx>(host, platform, Scalar{});
    alpaka::memset(queue, value, 0x00);

    auto status = alpaka::allocMappedBuf<bool, Idx>(host, platform, Scalar{});
    alpaka::memset(queue, status, 0xff);

    auto const extent = alpaka::Vec<Dim, Idx>::all(32);
    auto const elems = alpaka::Vec<Dim, Idx>::all(4);

    KernelOncePerBlock kernel;
    alpaka::KernelCfg<Acc> const config = {extent, elems, false};
    auto const workDiv = alpaka::getValidWorkDiv(config, device, kernel, std::data(status), std::data(value));
    const int32_t blocks = static_cast<int32_t>(alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(workDiv).prod());

    alpaka::exec<Acc>(queue, workDiv, kernel, std::data(status), std::data(value));
    alpaka::wait(queue);

    fixBooleanValue(*status);
    REQUIRE(*status == true);
    REQUIRE(*value == blocks);
}
