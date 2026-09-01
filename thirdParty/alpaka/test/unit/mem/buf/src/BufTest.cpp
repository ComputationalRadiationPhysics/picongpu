/* Copyright 2025 Axel Huebl, Benjamin Worpitz, Andrea Bocci, Bernhard Manfred Gruber, Jeffrey Kelling, Jan Stephan,
 *                Aurora Perego, Simone Balducci
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
    template<typename TDim, typename TDev, typename TElem, typename TIdx, typename TExtent>
    auto allocBuf(TDev dev, TExtent extent)
    {
        return alpaka::allocBuf<TElem, TIdx>(dev, extent);
    }

    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocBuf(alpaka::DevCpu dev, TExtent extent) -> alpaka::BufCpu<TElem, TDim, TIdx>
    {
        return alpaka::allocBuf<TElem, TIdx>(dev, extent);
    }

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocBuf(alpaka::DevCudaRt dev, TExtent extent) -> alpaka::BufCudaRt<TElem, TDim, TIdx>
    {
        return alpaka::allocBuf<TElem, TIdx>(dev, extent);
    }
#endif
#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocBuf(alpaka::DevHipRt dev, TExtent extent) -> alpaka::BufHipRt<TElem, TDim, TIdx>
    {
        return alpaka::allocBuf<TElem, TIdx>(dev, extent);
    }
#endif
#if defined(ALPAKA_ACC_SYCL_ENABLED) and defined(ALPAKA_SYCL_ONEAPI_CPU)
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocBuf(alpaka::DevCpuSycl dev, TExtent extent) -> alpaka::BufCpuSycl<TElem, TDim, TIdx>
    {
        return alpaka::allocBuf<TElem, TIdx>(dev, extent);
    }
#endif
#if defined(ALPAKA_ACC_SYCL_ENABLED) and defined(ALPAKA_SYCL_ONEAPI_GPU)
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocBuf(alpaka::DevGpuSyclIntel dev, TExtent extent) -> alpaka::BufGpuSyclIntel<TElem, TDim, TIdx>
    {
        return alpaka::allocBuf<TElem, TIdx>(dev, extent);
    }
#endif
#if defined(ALPAKA_ACC_SYCL_ENABLED) and defined(ALPAKA_SYCL_ONEAPI_GPU_NVIDIA)
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocBuf(alpaka::DevGpuSyclNvidia dev, TExtent extent) -> alpaka::BufGpuSyclNvidia<TElem, TDim, TIdx>
    {
        return alpaka::allocBuf<TElem, TIdx>(dev, extent);
    }
#endif
#if defined(ALPAKA_ACC_SYCL_ENABLED) and defined(ALPAKA_SYCL_ONEAPI_GPU_AMD)
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocBuf(alpaka::DevGpuSyclAmd dev, TExtent extent) -> alpaka::BufGpuSyclAmd<TElem, TDim, TIdx>
    {
        return alpaka::allocBuf<TElem, TIdx>(dev, extent);
    }
#endif
#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_FPGA)
    template<typename TDim, typename TElem, typename TIdx, typename TExtent>
    auto allocBuf(alpaka::DevFpgaSyclIntel dev, TExtent extent) -> alpaka::BufFpgaSyclIntel<TElem, TDim, TIdx>
    {
        return alpaka::allocBuf<TElem, TIdx>(dev, extent);
    }
#endif

} // namespace buftest

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
    using Platform = alpaka::Platform<TAcc>;
    if constexpr(
        std::is_same_v<Platform, alpaka::PlatformCpu> or std::is_same_v<alpaka::AccToTag<TAcc>, alpaka::TagCpuSycl>)
    {
        INFO("element " << index << " returned at offset " << reinterpret_cast<uintptr_t>(&buf[index]) - base);

        CHECK(reinterpret_cast<Elem*>(expected) == &buf[index]);

        // check that an out-of-bound access is detected
        if constexpr(Dim::value > 0)
            CHECK_THROWS_AS((void) buf.at(extent), std::out_of_range);
    }
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
    using DimExtent = alpaka::DimInt<0>;

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platformAcc, 0);
    auto queue = alpaka::Queue<Acc, alpaka::Blocking>{dev};
    auto const extent = alpaka::Vec<DimExtent, Idx>{};

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
        auto buf1 = buftest::allocBuf<DimExtent, Elem, Idx>(dev, extent);
        write(buf1, 1);
        auto buf2{std::move(buf1)};
        CHECK(read(buf2) == 1);
    } // both buffers destruct fine here

    // move assignment (via swap)
    {
        auto buf1 = buftest::allocBuf<DimExtent, Elem, Idx>(dev, extent);
        auto buf2 = buftest::allocBuf<DimExtent, Elem, Idx>(dev, extent);
        write(buf1, 1);
        write(buf2, 2);
        using std::swap;
        swap(buf1, buf2);
        CHECK(read(buf1) == 2);
        CHECK(read(buf2) == 1);
    } // both buffers destruct fine here
}

namespace
{

    struct DummyType
    {
        auto check() -> bool
        {
            return true;
        }
    };

} // namespace

TEMPLATE_LIST_TEST_CASE("memBufAccessors", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    using Elem = std::size_t;
    using Dim = alpaka::Dim<Acc>;

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platformAcc, 0);

    // accessors for scalar buffers
    {
        auto buf = buftest::allocBuf<alpaka::DimInt<0u>, Elem, Idx>(devHost, alpaka::Vec<alpaka::DimInt<0u>, Idx>{});
        *buf = 42u;
        CHECK(*buf == 42);

        auto buf2
            = buftest::allocBuf<alpaka::DimInt<0u>, DummyType, Idx>(devHost, alpaka::Vec<alpaka::DimInt<0u>, Idx>{});
        CHECK(buf2->check());
    }

    if constexpr(Dim::value == 1)
    {
        auto queue = alpaka::Queue<Acc, alpaka::Blocking>{dev};
        auto const extent = alpaka::Vec<Dim, Idx>{};
        auto foo = [](std::span<Elem>) { return true; };

        auto buf = buftest::allocBuf<Dim, Elem, Idx>(dev, extent);

        CHECK(buf.size() == extent[0]);
        CHECK(buf.begin() == buf.data());
        CHECK(buf.cbegin() == buf.data());
        CHECK(buf.end() == buf.data() + buf.size());
        CHECK(buf.cend() == buf.data() + buf.size());
        CHECK((buf.end() - buf.begin()) == static_cast<long>(buf.size()));
        CHECK((buf.cend() - buf.cbegin()) == static_cast<long>(buf.size()));
        CHECK(foo(buf));
    }
}

TEMPLATE_LIST_TEST_CASE("memBufAllocDeducedIdx", "[memBuf]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    using Elem = std::size_t;
    using Dim = alpaka::Dim<Acc>;
    auto const extent = alpaka::Vec<Dim, Idx>{};
    auto const devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
    auto const devAcc = alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0);
    auto queue = alpaka::Queue<Acc, alpaka::Blocking>{devAcc};

    auto host_buf = alpaka::allocBuf<Elem>(devHost, extent);
    auto dev_buf = alpaka::allocBuf<Elem>(devAcc, extent);
    auto dev_buf_async = alpaka::allocAsyncBuf<Elem>(queue, extent);
    auto dev_buf_async_supported = alpaka::allocAsyncBufIfSupported<Elem>(queue, extent);
    auto buf_mapped = alpaka::allocMappedBuf<Elem>(devHost, alpaka::PlatformCpu{}, extent);
    auto buf_managed = alpaka::allocManagedBuf<Elem>(devHost, alpaka::PlatformCpu{}, extent);
}
