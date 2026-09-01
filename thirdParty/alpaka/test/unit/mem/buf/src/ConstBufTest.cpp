/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <type_traits>

namespace buftest
{
    template<typename TDev, typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto initConstBuf(TDev dev, TExtent extent)
    {
        using QueueAcc = alpaka::test::DefaultQueue<TDev>;

        // allocate buffer
        auto buf = alpaka::allocBuf<TElem, TIdx>(dev, extent);

        // fill buffer
        auto const platformHost = alpaka::PlatformCpu{};
        auto const devHost = alpaka::getDevByIdx(platformHost, 0);
        std::vector<TElem> dataHost(static_cast<std::size_t>(getExtentProduct(extent)), static_cast<TElem>(1));
        auto bufHost = alpaka::createView(devHost, dataHost.data(), extent);
        QueueAcc queueAcc{dev};
        alpaka::memcpy(queueAcc, buf, bufHost);
        alpaka::wait(queueAcc);

#if ALPAKA_COMP_CLANG >= ALPAKA_VERSION_NUMBER(21, 1, 0)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wnrvo"
#endif

        // make the buffer constant and return only that
        auto const constBuf = alpaka::makeConstBuf(std::move(buf));
        return constBuf;

#if ALPAKA_COMP_CLANG >= ALPAKA_VERSION_NUMBER(21, 1, 0)
#    pragma clang diagnostic pop
#endif
    }

    template<typename TBuf>
    concept onlyConstNativePtr = requires(TBuf t) {
        {
            alpaka::getPtrNative(t)
        } -> std::same_as<alpaka::Elem<std::remove_const_t<TBuf>> const*>;
    };
} // namespace buftest

template<typename TAcc>
static auto testConstBuffer(alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& extent) -> void
{
    using Dev = alpaka::Dev<TAcc>;

    using Elem = float;
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;

    auto const platformAcc = alpaka::Platform<TAcc>{};
    auto const dev = alpaka::getDevByIdx(platformAcc, 0);

    // alpaka::malloc
    auto buf = alpaka::allocBuf<Elem, Idx>(dev, extent);
    auto const c_buf = buftest::initConstBuf<Dev, Dim, Elem, Idx>(dev, extent);

    using TBuf = decltype(buf);
    using TCBuf = decltype(c_buf);

    auto const offset = alpaka::Vec<Dim, Idx>::zeros();
    alpaka::test::testViewImmutable<Elem const>(c_buf, dev, extent, offset);

    // check that Constant buffer can't be converted to non-const buffer
    STATIC_REQUIRE_FALSE(std::convertible_to<TCBuf, TBuf>);
    STATIC_REQUIRE(std::convertible_to<TCBuf, TCBuf>);
    STATIC_REQUIRE(std::convertible_to<TBuf, TBuf>);
    STATIC_REQUIRE(std::convertible_to<TBuf, TCBuf>);

    STATIC_REQUIRE_FALSE(buftest::onlyConstNativePtr<TBuf>);
    STATIC_REQUIRE(buftest::onlyConstNativePtr<TCBuf>);
    // *getPtrNative(c_buf) = 0.f;  // <- this does not compile, as desired

    // check return types of the buffers
    using Platform = alpaka::Platform<TAcc>;
    if constexpr(
        std::is_same_v<Platform, alpaka::PlatformCpu> or std::is_same_v<alpaka::AccToTag<TAcc>, alpaka::TagCpuSycl>)
    {
        STATIC_REQUIRE(std::is_same_v<decltype(buf[0]), Elem&>);
        STATIC_REQUIRE(std::is_same_v<decltype(c_buf[0]), Elem const&>);
    }

    // check movability construction of buffers
    STATIC_REQUIRE(std::movable<TBuf>);
    STATIC_REQUIRE(std::movable<std::remove_const_t<TCBuf>>);
}

template<typename TAcc>
static auto testConstBufLifetime(alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& extent) -> void
{
    using Dev = alpaka::Dev<TAcc>;
    using Queue = alpaka::test::DefaultQueue<Dev>;

    using Elem = std::uint32_t;
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;

    auto const platformAcc = alpaka::Platform<TAcc>{};
    auto const dev = alpaka::getDevByIdx(platformAcc, 0);

    // init and return a const buffer filled with all ones
    auto const c_buf = buftest::initConstBuf<Dev, Dim, Elem, Idx>(dev, extent);

    Queue queue{dev};

    // create local buffer filled with all ones
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    std::vector<Elem> dataHost(static_cast<std::size_t>(getExtentProduct(extent)), static_cast<Elem>(1));
    auto bufHost = alpaka::createView(devHost, dataHost.data(), extent);

    auto copiedBack = alpaka::allocBuf<Elem, Idx>(devHost, extent);
    alpaka::memcpy(queue, copiedBack, c_buf);
    alpaka::wait(queue);

    bool resultCorrect = true;
    auto const pHostData = std::data(bufHost);
    auto const pCopiedBackData = std::data(copiedBack);
    for(Idx i(0u); i < getExtentProduct(extent); ++i)
    {
        if(pHostData[i] != pCopiedBackData[i])
        {
            resultCorrect = false;
        }
    }

    REQUIRE(resultCorrect);
}

TEMPLATE_LIST_TEST_CASE("constMemBufBasicTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    testConstBuffer<Acc>(alpaka::test::extentBuf<Dim, Idx>);
    testConstBufLifetime<Acc>(alpaka::test::extentBuf<Dim, Idx>);
}
