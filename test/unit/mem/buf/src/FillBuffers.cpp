/* Copyright 2025 Maria Michailidi, Anna Polova, Abdulrahman Al Marzouqi
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <type_traits>

TEMPLATE_LIST_TEST_CASE("memBufFillPrimitiveValuesTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dev = alpaka::Dev<Acc>;
    using Queue = alpaka::test::DefaultQueue<Dev>;
    using Elem = int;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platformAcc, 0);

    INFO("Test fill function on device");
    INFO(alpaka::getName(dev));

    Queue queue(dev);

    auto const extent = alpaka::test::extentBuf<Dim, Idx>;

    auto buf = alpaka::allocBuf<Elem, Idx>(dev, extent);

    constexpr Elem fillVal = 42;
    alpaka::fill(queue, buf, fillVal);

    // Copy result to host and check
    auto bufHost = alpaka::allocBuf<Elem, Idx>(devHost, extent);
    alpaka::memcpy(queue, bufHost, buf);
    alpaka::wait(queue);

    Elem const* ptr = std::data(bufHost);
    Idx const size = alpaka::getExtentProduct(bufHost);
    bool passed = true;
    for(Idx i = 0; i < size; ++i)
    {
        if(ptr[i] != fillVal)
        {
            passed = false;
        }
    }
    CHECK(passed);
}

struct Elem
{
    int i;
    float f;

    bool operator==(Elem const& other) const
    {
        return i == other.i && std::fabs(f - other.f) < 1e-9f;
    }
};

TEMPLATE_LIST_TEST_CASE("memBufFillNonPrimitiveValuesTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dev = alpaka::Dev<Acc>;
    using Queue = alpaka::test::DefaultQueue<Dev>;

    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platformAcc, 0);

    INFO("Test fill function on device");
    INFO(alpaka::getName(dev));

    Queue queue(dev);

    auto const extent = alpaka::test::extentBuf<Dim, Idx>;

    auto buf = alpaka::allocBuf<Elem, Idx>(dev, extent);

    constexpr Elem fillVal = {42, 99.0f};
    alpaka::fill(queue, buf, fillVal);

    // Copy result to host and check
    auto bufHost = alpaka::allocBuf<Elem, Idx>(devHost, extent);
    alpaka::memcpy(queue, bufHost, buf);
    alpaka::wait(queue);

    Elem const* ptr = std::data(bufHost);
    Idx const size = alpaka::getExtentProduct(bufHost);
    bool passed = true;
    for(Idx i = 0; i < size; ++i)
    {
        if(ptr[i] != fillVal)
        {
            passed = false;
        }
    }
    CHECK(passed);
}

TEMPLATE_LIST_TEST_CASE("memBufFillScalarFloatTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dev = alpaka::Dev<Acc>;
    using Queue = alpaka::test::DefaultQueue<Dev>;
    using ElemT = float;
    using Idx = alpaka::Idx<Acc>;

    float epsilon = 1e-9f;

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platformAcc, 0);

    INFO("Test fill function on device");
    INFO(alpaka::getName(dev));

    Queue queue(dev);

    auto const extent = alpaka::test::extentBuf<alpaka::DimInt<0u>, Idx>;

    auto buf = alpaka::allocBuf<ElemT, Idx>(dev, extent);

    constexpr ElemT fillVal = 42.0f;
    alpaka::fill(queue, buf, fillVal);

    // Copy result to host and check
    auto bufHost = alpaka::allocBuf<ElemT, Idx>(devHost, extent);
    alpaka::memcpy(queue, bufHost, buf);
    alpaka::wait(queue);

    ElemT const* ptr = std::data(bufHost);
    CHECK(std::fabs(ptr[0] - fillVal) < epsilon);
}
