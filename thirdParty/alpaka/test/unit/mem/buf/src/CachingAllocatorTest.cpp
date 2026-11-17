/* Copyright 2026 Simone Balducci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)

#    include <cub/util_allocator.cuh>

#    include <stdexcept>

namespace
{
    //! A minimal wrapper adapting cub::CachingDeviceAllocator to alpaka's alpaka::concepts::Allocator
    //! interface. The blocks are associated with the given stream, so that cub takes care of ordering their reuse
    //! with respect to the work enqueued on it.
    class AllocatorWrapper
    {
    private:
        cub::CachingDeviceAllocator* m_allocator;
        cudaStream_t m_stream;

    public:
        AllocatorWrapper(cub::CachingDeviceAllocator* allocator, cudaStream_t stream)
            : m_allocator(allocator)
            , m_stream(stream)
        {
        }

        auto allocate(std::size_t bytes, std::size_t align) -> void*
        {
            void* ptr = nullptr;
            if(m_allocator->DeviceAllocate(&ptr, bytes, m_stream) != cudaSuccess)
                throw std::runtime_error("DeviceAllocate failed");
            if(reinterpret_cast<std::uintptr_t>(ptr) % align != 0)
                throw std::runtime_error("DeviceAllocate returned unaligned pointer");
            return ptr;
        }

        auto deallocate(void* ptr) -> void
        {
            if(m_allocator->DeviceFree(ptr) != cudaSuccess)
                throw std::runtime_error("DeviceFree failed");
        }
    };

    static_assert(alpaka::concepts::Allocator<AllocatorWrapper>);
} // namespace

TEST_CASE("cachingAllocatorBufAllocTest", "[memBuf][cachingAllocator]")
{
    using Acc = alpaka::AccGpuCudaRt<alpaka::DimInt<1u>, std::size_t>;
    using Dev = alpaka::Dev<Acc>;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    using Elem = float;
    using Queue = alpaka::test::DefaultQueue<Dev>;

    auto dev = alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0u);
    Queue queue(dev);

    cub::CachingDeviceAllocator cubAllocator;
    AllocatorWrapper allocator(&cubAllocator, alpaka::getNativeHandle(queue));

    auto const extent = alpaka::test::extentBuf<Dim, Idx>;
    auto buf = alpaka::allocBuf<Elem, Idx>(dev, extent, allocator);

    auto const offset = alpaka::Vec<Dim, Idx>::zeros();
    alpaka::test::testViewImmutable<Elem>(buf, dev, extent, offset);
    alpaka::test::testViewMutable<Acc>(queue, buf);
}

TEST_CASE("cachingAllocatorAsyncBufAllocTest", "[memBuf][cachingAllocator]")
{
    using Acc = alpaka::AccGpuCudaRt<alpaka::DimInt<1u>, std::size_t>;
    using Dev = alpaka::Dev<Acc>;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    using Elem = float;
    using Queue = alpaka::test::DefaultQueue<Dev>;

    auto dev = alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0u);
    Queue queue(dev);

    cub::CachingDeviceAllocator cubAllocator;
    AllocatorWrapper allocator(&cubAllocator, alpaka::getNativeHandle(queue));

    auto const extent = alpaka::test::extentBuf<Dim, Idx>;

    STATIC_REQUIRE(alpaka::hasAsyncBufSupport<Dev, Dim>);

    auto buf = alpaka::allocAsyncBuf<Elem, Idx>(queue, extent, allocator);

    alpaka::test::testViewMutable<Acc>(queue, buf);

    alpaka::wait(queue);
    auto const offset = alpaka::Vec<Dim, Idx>::zeros();
    alpaka::test::testViewImmutable<Elem>(buf, dev, extent, offset);
}

TEST_CASE("cachingAllocatorReusesFreedMemoryTest", "[memBuf][cachingAllocator]")
{
    using Acc = alpaka::AccGpuCudaRt<alpaka::DimInt<1u>, std::size_t>;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    using Elem = float;
    using Queue = alpaka::test::DefaultQueue<alpaka::Dev<Acc>>;

    auto dev = alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0u);
    Queue queue(dev);

    cub::CachingDeviceAllocator cubAllocator;
    AllocatorWrapper allocator(&cubAllocator, alpaka::getNativeHandle(queue));

    auto const extent = alpaka::test::extentBuf<Dim, Idx>;

    void const* firstPtr = nullptr;
    {
        auto buf = alpaka::allocBuf<Elem, Idx>(dev, extent, allocator);
        firstPtr = std::data(buf);
    }

    auto buf = alpaka::allocBuf<Elem, Idx>(dev, extent, allocator);
    CHECK(firstPtr == std::data(buf));
}

#endif
