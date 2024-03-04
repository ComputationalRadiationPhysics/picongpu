/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Andrea Bocci, Bernhard Manfred Gruber, Jeffrey Kelling, Jan Stephan
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

template<typename TAcc>
static auto testBufferMutable(alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& extent) -> void
{
    using Dev = alpaka::Dev<TAcc>;
    using Queue = alpaka::test::DefaultQueue<Dev>;

    using Elem = float;
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;

    auto const platformAcc = alpaka::Platform<TAcc>{};
    auto const dev = alpaka::getDevByIdx(platformAcc, 0);
    Queue queue(dev);

    // alpaka::malloc
    auto buf = alpaka::allocBuf<Elem, Idx>(dev, extent);

    auto const offset = alpaka::Vec<Dim, Idx>::zeros();
    alpaka::test::testViewImmutable<Elem>(buf, dev, extent, offset);

    alpaka::test::testViewMutable<TAcc>(queue, buf);
}

template<typename TAcc>
static auto testAsyncBufferMutable(alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& extent) -> void
{
    using Dev = alpaka::Dev<TAcc>;
    using Queue = alpaka::test::DefaultQueue<Dev>;

    using Elem = float;
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;

    auto const platformAcc = alpaka::Platform<TAcc>{};
    auto const dev = alpaka::getDevByIdx(platformAcc, 0);
    Queue queue(dev);

    // memory is allocated when the queue reaches this point
    auto buf = alpaka::allocAsyncBuf<Elem, Idx>(queue, extent);

    // asynchronous operations can be submitted to the queue immediately
    alpaka::test::testViewMutable<TAcc>(queue, buf);

    // synchronous operations must wait for the memory to be available
    alpaka::wait(queue);
    auto const offset = alpaka::Vec<Dim, Idx>::zeros();
    alpaka::test::testViewImmutable<Elem>(buf, dev, extent, offset);

    // the buffer will queue the deallocation of the memory when it goes out of scope,
    // and extend the lifetime of the queue until all memory operations have completed.
}

TEMPLATE_LIST_TEST_CASE("memBufBasicTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    testBufferMutable<Acc>(alpaka::test::extentBuf<Dim, Idx>);
}

TEMPLATE_LIST_TEST_CASE("memBufZeroSizeTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    auto const extent = alpaka::Vec<Dim, Idx>::zeros();

    testBufferMutable<Acc>(extent);
}

TEMPLATE_LIST_TEST_CASE("memBufAsyncBasicTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    if constexpr(alpaka::hasAsyncBufSupport<alpaka::Dev<Acc>, Dim>)
    {
        testAsyncBufferMutable<Acc>(alpaka::test::extentBuf<Dim, Idx>);
    }
    else
    {
        INFO("Stream-ordered memory buffers are not supported in this configuration.");
    }
}

TEMPLATE_LIST_TEST_CASE("memBufAsyncZeroSizeTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    if constexpr(alpaka::hasAsyncBufSupport<alpaka::Dev<Acc>, Dim>)
    {
        auto const extent = alpaka::Vec<Dim, Idx>::zeros();
        testAsyncBufferMutable<Acc>(extent);
    }
    else
    {
        INFO("Stream-ordered memory buffers are not supported in this configuration.");
    }
}

template<typename TAcc>
static auto testBufferImmutable(alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& extent) -> void
{
    using Elem = float;
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;

    auto const platformAcc = alpaka::Platform<TAcc>{};
    auto const dev = alpaka::getDevByIdx(platformAcc, 0);

    // alpaka::malloc
    auto const buf = alpaka::allocBuf<Elem, Idx>(dev, extent);

    auto const offset = alpaka::Vec<Dim, Idx>::zeros();
    alpaka::test::testViewImmutable<Elem>(buf, dev, extent, offset);
}

TEMPLATE_LIST_TEST_CASE("memBufConstTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    testBufferImmutable<Acc>(alpaka::test::extentBuf<Dim, Idx>);
}

template<typename TAcc>
static auto testAsyncBufferImmutable(alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& extent) -> void
{
    {
        using Dev = alpaka::Dev<TAcc>;
        using Queue = alpaka::test::DefaultQueue<Dev>;

        using Elem = float;
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;

        auto const platformAcc = alpaka::Platform<TAcc>{};
        auto const dev = alpaka::getDevByIdx(platformAcc, 0);
        Queue queue(dev);

        // memory is allocated when the queue reaches this point
        auto const buf = alpaka::allocAsyncBuf<Elem, Idx>(queue, extent);

        // synchronous operations must wait for the memory to be available
        alpaka::wait(queue);
        auto const offset = alpaka::Vec<Dim, Idx>::zeros();
        alpaka::test::testViewImmutable<Elem>(buf, dev, extent, offset);

        // The buffer will queue the deallocation of the memory when it goes out of scope,
        // and extend the lifetime of the queue until all memory operations have completed.
        // Delay the end of the queue to push the buffer deletetion task after
        // all local refs to queue have been dropped.
        alpaka::enqueue(queue, []() { std::this_thread::sleep_for(std::chrono::microseconds(1000)); });
        [](auto) {}(std::move(queue));
    }

    // Give the queue, including the buffer's deleter time to complete, we
    // cannot synchronize here because we dropped the handle to see how it
    // behaves when it self-destructs.
    std::this_thread::sleep_for(std::chrono::microseconds(1200));
}

TEMPLATE_LIST_TEST_CASE("memBufAsyncConstTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    if constexpr(alpaka::hasAsyncBufSupport<alpaka::Dev<Acc>, Dim>)
    {
        testAsyncBufferImmutable<Acc>(alpaka::test::extentBuf<Dim, Idx>);
    }
    else
    {
        INFO("Stream-ordered memory buffers are not supported in this configuration.");
    }
}

template<typename TAcc>
static auto testBufferAccessorAdaptor(
    alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& extent,
    alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> const& index) -> void
{
    using Elem = float;
    using Dim = alpaka::Dim<TAcc>;
    using Idx = alpaka::Idx<TAcc>;

    // assume dimensionality up to 4
    CHECK(Dim::value <= 4);

    auto const platformAcc = alpaka::Platform<TAcc>{};
    auto const dev = alpaka::getDevByIdx(platformAcc, 0);

    auto buf = alpaka::allocBuf<Elem, Idx>(dev, extent);

    // check that the array subscript operator access the correct element
    auto const& pitch = alpaka::getPitchesInBytes(buf);
    INFO("buffer extent: " << extent << " elements");
    INFO("buffer pitch: " << pitch << " bytes");
    CHECK((index < extent).all());

    auto const base = reinterpret_cast<uintptr_t>(std::data(buf));
    auto const expected = base + static_cast<uintptr_t>((pitch * index).sum());
    INFO("element " << index << " expected at offset " << expected - base);
    INFO("element " << index << " returned at offset " << reinterpret_cast<uintptr_t>(&buf[index]) - base);
    CHECK(reinterpret_cast<Elem*>(expected) == &buf[index]);

    // check that an out-of-bound access is detected
    if constexpr(Dim::value > 0)
        CHECK_THROWS_AS((void) buf.at(extent), std::out_of_range);
}

TEMPLATE_LIST_TEST_CASE("memBufAccessorAdaptorTest", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    testBufferAccessorAdaptor<Acc>(alpaka::test::extentBuf<Dim, Idx>, alpaka::test::offset<Dim, Idx>);
}

TEMPLATE_LIST_TEST_CASE("memBufMove", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    using Elem = std::size_t;

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platformAcc, 0);
    auto queue = alpaka::Queue<Acc, alpaka::Blocking>{dev};
    auto const extent = alpaka::Vec<alpaka::DimInt<0>, Idx>{};

    auto write = [&](auto& buf, Elem value)
    {
        auto v = alpaka::createView(devHost, &value, extent);
        alpaka::memcpy(queue, buf, v);
    };
    auto read = [&](auto const& buf)
    {
        Elem value{};
        auto v = alpaka::createView(devHost, &value, extent);
        alpaka::memcpy(queue, v, buf);
        return value;
    };

    // move constructor
    {
        auto buf1 = alpaka::allocBuf<Elem, Idx>(dev, extent);
        write(buf1, 1);
        auto buf2{std::move(buf1)};
        CHECK(read(buf2) == 1);
    } // both buffers destruct fine here

    // move assignment (via swap)
    {
        auto buf1 = alpaka::allocBuf<Elem, Idx>(dev, extent);
        auto buf2 = alpaka::allocBuf<Elem, Idx>(dev, extent);
        write(buf1, 1);
        write(buf2, 2);
        using std::swap;
        swap(buf1, buf2);
        CHECK(read(buf1) == 2);
        CHECK(read(buf2) == 1);
    } // both buffers destruct fine here
}
