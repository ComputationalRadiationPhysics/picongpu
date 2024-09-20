/* Copyright 2024 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include "alpaka/exec/UniformElements.hpp"

#include "WorkDiv.hpp"
#include "alpaka/acc/Traits.hpp"
#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/exec/Once.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/platform/PlatformCpu.hpp"
#include "alpaka/platform/Traits.hpp"
#include "alpaka/queue/Traits.hpp"
#include "alpaka/test/acc/TestAccs.hpp"
#include "alpaka/wait/Traits.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>
#include <type_traits>

#if BOOST_COMP_MSVC
// MSVC uses __restrict instead of __restrict__.
#    define __restrict__ __restrict
#endif

#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif
// Global Host object used by all tests.
using Host = alpaka::DevCpu;
static Host host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif

struct VectorAddKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        T const* __restrict__ in1,
        T const* __restrict__ in2,
        T* __restrict__ out,
        alpaka::Idx<TAcc> size) const
    {
        for(auto index : alpaka::uniformElements(acc, size))
        {
            out[index] = in1[index] + in2[index];
        }
    }
};

struct VectorAddKernelSkip
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        T const* __restrict__ in1,
        T const* __restrict__ in2,
        T* __restrict__ out,
        alpaka::Idx<TAcc> first,
        alpaka::Idx<TAcc> size) const
    {
        for(auto index : alpaka::uniformElements(acc, first, size))
        {
            out[index] = in1[index] + in2[index];
        }
    }
};

struct VectorAddKernel1D
{
    template<typename TAcc, typename T, typename = std::enable_if_t<alpaka::Dim<TAcc>::value == 1u>>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        T const* __restrict__ in1,
        T const* __restrict__ in2,
        T* __restrict__ out,
        alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> size) const
    {
        for(auto ndindex : alpaka::uniformElementsND(acc, size))
        {
            auto index = ndindex[0];
            out[index] = in1[index] + in2[index];
        }
    }
};

struct VectorAddKernel2D
{
    template<typename TAcc, typename T, typename = std::enable_if_t<alpaka::Dim<TAcc>::value == 2u>>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        T const* __restrict__ in1,
        T const* __restrict__ in2,
        T* __restrict__ out,
        alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> size) const
    {
        for(auto ndindex : alpaka::uniformElementsND(acc, size))
        {
            auto index = ndindex[0] * size[1] + ndindex[1];
            out[index] = in1[index] + in2[index];
        }
    }
};

struct VectorAddKernel3D
{
    template<typename TAcc, typename T, typename = std::enable_if_t<alpaka::Dim<TAcc>::value == 3u>>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        T const* __restrict__ in1,
        T const* __restrict__ in2,
        T* __restrict__ out,
        alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> size) const
    {
        for(auto ndindex : alpaka::uniformElementsND(acc, size))
        {
            auto index = (ndindex[0] * size[1] + ndindex[1]) * size[2] + ndindex[2];
            out[index] = in1[index] + in2[index];
        }
    }
};

/* This is not an efficient approach, and it uses more operations and synchronisations than needed. It is written like
 * this to test the use of dynamic shared memory, split block and element loops, and block-level synchronisations.
 */

struct VectorAddBlockKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        T const* __restrict__ in1,
        T const* __restrict__ in2,
        T* __restrict__ out,
        alpaka::Idx<TAcc> size) const
    {
        using Idx = alpaka::Idx<TAcc>;

        // Get the block size.
        auto const blockSize = alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u];

        // Get the dynamic shared memory buffer.
        T* buffer = alpaka::getDynSharedMem<T>(acc);

        // Split the loop over the elements into an outer loop over the groups and an inner loop over the elements, to
        // permit the synchronisation of the threads in each block between each step: the outer loop is needed to
        // repeat the "block" as many times as needed to cover the whole problem space; the inner loop is needed for
        // work division with more than one element per thread.
        for(auto block : alpaka::uniformGroups(acc, size))
        {
            // Initialise the shared memory.
            if(alpaka::oncePerBlock(acc))
            {
                for(Idx local = 0; local < blockSize; ++local)
                {
                    buffer[local] = 0;
                }
            }
            // Synchronise all threads in the block.
            alpaka::syncBlockThreads(acc);
            // Accumulate the first set of data into shared memory.
            for(auto index : alpaka::uniformGroupElements(acc, block, size))
            {
                buffer[index.local] += in1[index.global];
            }
            // Synchronise all threads in the block.
            alpaka::syncBlockThreads(acc);
            // Accumulate the second set of data into shared memory.
            for(auto index : alpaka::uniformGroupElements(acc, block, size))
            {
                buffer[index.local] += in2[index.global];
            }
            // Synchronise all threads in the block.
            alpaka::syncBlockThreads(acc);
            // Store the results into global memory.
            for(auto index : alpaka::uniformGroupElements(acc, block, size))
            {
                out[index.global] = buffer[index.local];
            }
            // Synchronise all threads in the block; this is necessary to avoid race conditions between different
            // iterations of the uniformGroups loop.
            alpaka::syncBlockThreads(acc);
        }
    }
};

/* Run all operations in a single thread.
 * Written in an inefficient way to test "oncePerGrid".
 */

struct VectorAddKernelSerial
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        T const* __restrict__ in1,
        T const* __restrict__ in2,
        T* __restrict__ out,
        alpaka::Idx<TAcc> size) const
    {
        using Idx = alpaka::Idx<TAcc>;

        // The operations are performed by a single thread.
        if(alpaka::oncePerGrid(acc))
        {
            for(Idx index = 0; index < size; ++index)
            {
                // Unsafe, used to test that each element is summed exactly once.
                out[index] += in1[index];
                out[index] += in2[index];
            }
        }
    }
};

/* Run all operations in one thread per block.
 * Written in an inefficient way to test "oncePerBlock".
 */

struct VectorAddKernelBlockSerial
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        T const* __restrict__ in1,
        T const* __restrict__ in2,
        T* __restrict__ out,
        alpaka::Idx<TAcc> size) const
    {
        using Idx = alpaka::Idx<TAcc>;

        // Get the block size.
        auto const blockSize = alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u];
        // The loop is used to repeat the "block" as many times as needed to cover the whole problem space.
        for(auto block : alpaka::uniformGroups(acc, size))
        {
            // The operations are performed by a single thread in each "logical" block.
            auto const first = blockSize * block;
            auto const range = std::min<Idx>(first + blockSize, size);
            if(alpaka::oncePerBlock(acc))
            {
                for(Idx index = first; index < range; ++index)
                {
                    // Unsafe, used to test that each element is summed exactly once.
                    out[index] += in1[index];
                    out[index] += in2[index];
                }
            }
        }
    }
};

namespace alpaka::trait
{
    // Specialize the BlockSharedMemDynSizeBytes trait to specify the amount of block shared dynamic memory for the
    // VectorAddBlockKernel kernel.
    template<typename TAcc>
    struct BlockSharedMemDynSizeBytes<VectorAddBlockKernel, TAcc>
    {
        using Idx = alpaka::Idx<TAcc>;
        using Dim1D = alpaka::DimInt<1u>;
        using Vec1D = alpaka::Vec<Dim1D, Idx>;

        // The size in bytes of the shared memory allocated for a block.
        template<typename T>
        ALPAKA_FN_HOST_ACC static std::size_t getBlockSharedMemDynSizeBytes(
            VectorAddBlockKernel const& /* kernel */,
            Vec1D threads,
            Vec1D elements,
            T const* __restrict__ /* in1 */,
            T const* __restrict__ /* in2 */,
            T* __restrict__ /* out */,
            Idx /* size */)
        {
#if defined(__GNUC__)
            // Silence a potential warning about
            // warning: conversion to ‘long unsigned int’ from ‘long int’ may change the sign of the result
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
            return static_cast<std::size_t>(threads[0] * elements[0] * sizeof(T));
#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif
        }
    };
} // namespace alpaka::trait

// Test the 1-dimensional kernel on all devices.
template<typename TAcc, typename TKernel>
void testVectorAddKernel(
    alpaka::Idx<TAcc> problem_size,
    alpaka::Idx<TAcc> grid_size,
    alpaka::Idx<TAcc> block_size,
    TKernel kernel)
{
    using Acc = TAcc;
    using Idx = alpaka::Idx<Acc>;
    using Platform = alpaka::Platform<Acc>;
    using Device = alpaka::Dev<Platform>;
    using Queue = alpaka::Queue<Device, alpaka::NonBlocking>;

    // Random number generator with a gaussian distribution.
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0.f, 1.f};

    // Tolerance.
    constexpr float epsilon = 0.000001f;

    // Buffer size.
    const Idx size = problem_size;

    // Initialise the accelerator platform.
    Platform platform{};

    // Allocate input and output host buffers in pinned memory accessible by the Platform devices.
    auto in1_h = alpaka::allocMappedBuf<float, Idx>(host, platform, size);
    auto in2_h = alpaka::allocMappedBuf<float, Idx>(host, platform, size);
    auto out_h = alpaka::allocMappedBuf<float, Idx>(host, platform, size);

    // Fill the input buffers with random data, and the output buffer with zeros.
    for(Idx i = 0; i < size; ++i)
    {
        in1_h[i] = dist(rand);
        in2_h[i] = dist(rand);
        out_h[i] = 0.f;
    }

    // Run the test on each device.
    for(auto const& device : alpaka::getDevs(platform))
    {
        /* clang-format off */
        INFO("Test 1D vector addition on " << alpaka::getName(device) << " over " << problem_size << " values with "
                                           << grid_size << " blocks of " << block_size << " elements");
        /* clang-format on */
        auto queue = Queue(device);

        // Allocate input and output buffers on the device.
        auto in1_d = alpaka::allocAsyncBufIfSupported<float, Idx>(queue, size);
        auto in2_d = alpaka::allocAsyncBufIfSupported<float, Idx>(queue, size);
        auto out_d = alpaka::allocAsyncBufIfSupported<float, Idx>(queue, size);

        // Copy the input data to the device; the size is known from the buffer objects.
        alpaka::memcpy(queue, in1_d, in1_h);
        alpaka::memcpy(queue, in2_d, in2_h);

        // Fill the output buffer with zeros; the size is known from the buffer objects.
        alpaka::memset(queue, out_d, 0);

        // Launch the 1-dimensional kernel with scalar size.
        auto div = makeWorkDiv<Acc>(grid_size, block_size);
        alpaka::exec<Acc>(queue, div, kernel, in1_d.data(), in2_d.data(), out_d.data(), size);

        // Copy the results from the device to the host.
        alpaka::memcpy(queue, out_h, out_d);

        // Wait for all the operations to complete.
        alpaka::wait(queue);

        // Check the results.
        for(Idx i = 0; i < size; ++i)
        {
            float sum = in1_h[i] + in2_h[i];
            REQUIRE(out_h[i] < sum + epsilon);
            REQUIRE(out_h[i] > sum - epsilon);
        }
    }
}

// Test the 1-dimensional kernel on all devices, potentially skipping some elements.
template<typename TAcc, typename TKernel>
void testVectorAddKernelSkip(
    alpaka::Idx<TAcc> skip_elements,
    alpaka::Idx<TAcc> problem_size,
    alpaka::Idx<TAcc> grid_size,
    alpaka::Idx<TAcc> block_size,
    TKernel kernel)
{
    using Acc = TAcc;
    using Idx = alpaka::Idx<Acc>;
    using Platform = alpaka::Platform<Acc>;
    using Device = alpaka::Dev<Platform>;
    using Queue = alpaka::Queue<Device, alpaka::NonBlocking>;

    // Random number generator with a gaussian distribution.
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0.f, 1.f};

    // Tolerance.
    constexpr float epsilon = 0.000001f;

    // Buffer size.
    const Idx size = problem_size;

    // Initialise the accelerator platform.
    Platform platform{};

    // Allocate input and output host buffers in pinned memory accessible by the Platform devices.
    auto in1_h = alpaka::allocMappedBuf<float, Idx>(host, platform, size);
    auto in2_h = alpaka::allocMappedBuf<float, Idx>(host, platform, size);
    auto out_h = alpaka::allocMappedBuf<float, Idx>(host, platform, size);

    // Fill the input buffers with random data, and the output buffer with zeros.
    for(Idx i = 0; i < size; ++i)
    {
        in1_h[i] = dist(rand);
        in2_h[i] = dist(rand);
        out_h[i] = 0.f;
    }

    // Run the test on each device.
    for(auto const& device : alpaka::getDevs(platform))
    {
        /* clang-format off */
        INFO("Test 1D vector addition on " << alpaka::getName(device) << " skipping " << skip_elements << " over "
                                           << problem_size << " values with " << grid_size << " blocks of "
                                           << block_size << " elements");
        /* clang-format on */
        auto queue = Queue(device);

        // Allocate input and output buffers on the device.
        auto in1_d = alpaka::allocAsyncBufIfSupported<float, Idx>(queue, size);
        auto in2_d = alpaka::allocAsyncBufIfSupported<float, Idx>(queue, size);
        auto out_d = alpaka::allocAsyncBufIfSupported<float, Idx>(queue, size);

        // Copy the input data to the device; the size is known from the buffer objects.
        alpaka::memcpy(queue, in1_d, in1_h);
        alpaka::memcpy(queue, in2_d, in2_h);

        // Fill the output buffer with zeros; the size is known from the buffer objects.
        alpaka::memset(queue, out_d, 0);

        // Launch the 1-dimensional kernel with scalar size.
        auto div = makeWorkDiv<Acc>(grid_size, block_size);
        alpaka::exec<Acc>(queue, div, kernel, in1_d.data(), in2_d.data(), out_d.data(), skip_elements, size);

        // Copy the results from the device to the host.
        alpaka::memcpy(queue, out_h, out_d);

        // Wait for all the operations to complete.
        alpaka::wait(queue);

        // Check the results.
        for(Idx i = 0; i < skip_elements; ++i)
        {
            // The first part of the output vector should not have been modified at all, and should be identically 0.
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
            REQUIRE(out_h[i] == 0);
#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif
        }
        for(Idx i = skip_elements; i < size; ++i)
        {
            float sum = in1_h[i] + in2_h[i];
            REQUIRE(out_h[i] < sum + epsilon);
            REQUIRE(out_h[i] > sum - epsilon);
        }
    }
}

// Test the N-dimensional kernels on all devices.
template<typename TAcc, typename TKernel>
void testVectorAddKernelND(
    alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> problem_size,
    alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> grid_size,
    alpaka::Vec<alpaka::Dim<TAcc>, alpaka::Idx<TAcc>> block_size,
    TKernel kernel)
{
    using Acc = TAcc;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    using Platform = alpaka::Platform<Acc>;
    using Device = alpaka::Dev<Platform>;
    using Queue = alpaka::Queue<Device, alpaka::NonBlocking>;

    // Random number generator with a gaussian distribution.
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0.f, 1.f};

    // Tolerance.
    constexpr float epsilon = 0.000001f;

    // Linearised buffer size.
    const Idx size = problem_size.prod();

    // Initialise the accelerator platform.
    Platform platform{};

    // Allocate input and output host buffers in pinned memory accessible by the Platform devices.
    auto in1_h = alpaka::allocMappedBuf<float, Idx>(host, platform, size);
    auto in2_h = alpaka::allocMappedBuf<float, Idx>(host, platform, size);
    auto out_h = alpaka::allocMappedBuf<float, Idx>(host, platform, size);

    // Fill the input buffers with random data, and the output buffer with zeros.
    for(Idx i = 0; i < size; ++i)
    {
        in1_h[i] = dist(rand);
        in2_h[i] = dist(rand);
        out_h[i] = 0.f;
    }

    // Run the test on each device.
    for(auto const& device : alpaka::getDevs(platform))
    {
        /* clang-format off */
        INFO("Test " << Dim::value << "D vector addition on " << alpaka::getName(device) << " over " << problem_size
                     << " values with " << grid_size << " blocks of " << block_size << " elements");
        /* clang-format on */
        auto queue = Queue(device);

        // Allocate input and output buffers on the device.
        auto in1_d = alpaka::allocAsyncBufIfSupported<float, Idx>(queue, size);
        auto in2_d = alpaka::allocAsyncBufIfSupported<float, Idx>(queue, size);
        auto out_d = alpaka::allocAsyncBufIfSupported<float, Idx>(queue, size);

        // Copy the input data to the device; the size is known from the buffer objects.
        alpaka::memcpy(queue, in1_d, in1_h);
        alpaka::memcpy(queue, in2_d, in2_h);

        // Fill the output buffer with zeros; the size is known from the buffer objects.
        alpaka::memset(queue, out_d, 0);

        // Launch the 3-dimensional kernel.
        auto div = makeWorkDiv<Acc>(grid_size, block_size);
        alpaka::exec<Acc>(queue, div, kernel, in1_d.data(), in2_d.data(), out_d.data(), problem_size);

        // Copy the results from the device to the host.
        alpaka::memcpy(queue, out_h, out_d);

        // Wait for all the operations to complete.
        alpaka::wait(queue);

        // Check the results.
        for(Idx i = 0; i < size; ++i)
        {
            float sum = in1_h[i] + in2_h[i];
            REQUIRE(out_h[i] < sum + epsilon);
            REQUIRE(out_h[i] > sum - epsilon);
        }
    }
}

TEMPLATE_LIST_TEST_CASE("UniformElements", "[exec]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    using Vec = alpaka::Vec<Dim, Idx>;

    // 1-dimensional kernels.
    if constexpr(Dim::value == 1)
    {
        SECTION("VectorAddKernel1D, small block size")
        {
            // Launch the 1-dimensional kernel with a small block size and a small number of blocks; this relies on the
            // kernel to loop over the "problem space" and do more work per block.
            INFO("Test 1D vector addition with small block size");
            testVectorAddKernelND<TestType, VectorAddKernel1D>(Vec{10000}, Vec{32}, Vec{32}, VectorAddKernel1D{});
        }

        SECTION("VectorAddKernel1D, large block size")
        {
            // Launch the 1-dimensional kernel with a large block size and a single block; this relies on the kernel to
            // check the size of the "problem space" and avoid accessing out-of-bounds data.
            INFO("Test 1D vector addition with large block size");
            testVectorAddKernelND<TestType>({10}, {2}, {32}, VectorAddKernel1D{});
        }
    }

    // 2-dimensional kernels.
    if constexpr(Dim::value == 2)
    {
        SECTION("VectorAddKernel2D, small block size")
        {
            // Launch the 2-dimensional kernel with a small block size and a small number of blocks; this relies on the
            // kernel to loop over the "problem space" and do more work per block.
            INFO("Test 2D vector addition with small block size");
            testVectorAddKernelND<TestType>({400, 250}, {5, 4}, {8, 4}, VectorAddKernel2D{});
        }

        SECTION("VectorAddKernel2D, large block size")
        {
            // Launch the 2-dimensional kernel with a large block size and a single block; this relies on the kernel to
            // check the size of the "problem space" and avoid accessing out-of-bounds data.
            INFO("Test 2D vector addition with large block size");
            testVectorAddKernelND<TestType>({5, 3}, {2, 2}, {8, 4}, VectorAddKernel2D{});
        }
    }

    // 3-dimensional kernels.
    if constexpr(Dim::value == 3)
    {
        SECTION("VectorAddKernel3D, small block size")
        {
            // Launch the 3-dimensional kernel with a small block size and a small number of blocks; this relies on the
            // kernel to loop over the "problem space" and do more work per block.
            INFO("Test 3D vector addition with small block size");
            testVectorAddKernelND<TestType>({50, 25, 16}, {5, 2, 2}, {2, 4, 4}, VectorAddKernel3D{});
        }

        SECTION("VectorAddKernel3D, large block size")
        {
            // Launch the 3-dimensional kernel with a large block size and a single block; this relies on the kernel to
            // check the size of the "problem space" and avoid accessing out-of-bounds data.
            INFO("Test 3D vector addition with large block size");
            testVectorAddKernelND<TestType>({2, 3, 3}, {2, 2, 2}, {2, 4, 4}, VectorAddKernel3D{});
        }
    }

    // 1-dimensional kernels.
    if constexpr(Dim::value == 1)
    {
        SECTION("VectorAddKernel, small block size")
        {
            // Launch the 1-dimensional kernel with a small block size and a small number of blocks; this relies on the
            // kernel to loop over the "problem space" and do more work per block.
            INFO("Test 1D vector addition with small block size, using scalar dimensions");
            testVectorAddKernel<TestType>(10000, 32, 32, VectorAddKernel{});
        }

        SECTION("VectorAddKernel, large block size")
        {
            // Launch the 1-dimensional kernel with a large block size and a single block; this relies on the kernel to
            // check the size of the "problem space" and avoid accessing out-of-bounds data.
            INFO("Test 1D vector addition with large block size, using scalar dimensions");
            testVectorAddKernel<TestType>(10, 2, 32, VectorAddKernel{});
        }

        SECTION("VectorAddBlockKernel, small block size")
        {
            // Launch the 1-dimensional kernel with a small block size and a small number of blocks; this relies on the
            // kernel to loop over the "problem space" and do more work per block.
            INFO("Test 1D vector block-level addition with small block size, using scalar dimensions");
            testVectorAddKernel<TestType>(10000, 32, 32, VectorAddBlockKernel{});
        }

        SECTION("VectorAddBlockKernel, large block size")
        {
            // Launch the 1-dimensional kernel with a large block size and a single block; this relies on the kernel to
            // check the size of the "problem space" and avoid accessing out-of-bounds data.
            INFO("Test 1D vector block-level addition with large block size, using scalar dimensions");
            testVectorAddKernel<TestType>(10, 2, 32, VectorAddBlockKernel{});
        }

        SECTION("VectorAddKernelSerial, small block size")
        {
            // Launch the 1-dimensional kernel with a small block size and a small number of blocks; this relies on the
            // kernel to loop over the "problem space" and do more work per block.
            INFO("Test 1D vector single-threaded serial addition with small block size, using scalar dimensions");
            testVectorAddKernel<TestType>(10000, 32, 32, VectorAddKernelSerial{});
        }

        SECTION("VectorAddKernelSerial, large block size")
        {
            // Launch the 1-dimensional kernel with a large block size and a single block; this relies on the kernel to
            // check the size of the "problem space" and avoid accessing out-of-bounds data.
            INFO("Test 1D vector single-threaded seria addition with large block size, using scalar dimensions");
            testVectorAddKernel<TestType>(10, 2, 32, VectorAddKernelSerial{});
        }

        SECTION("VectorAddKernelBlockSerial, small block size")
        {
            // Launch the 1-dimensional kernel with a small block size and a small number of blocks; this relies on the
            // kernel to loop over the "problem space" and do more work per block.
            INFO("Test 1D vector block-level serial addition with small block size, using scalar dimensions");
            testVectorAddKernel<TestType>(10000, 32, 32, VectorAddKernelBlockSerial{});
        }

        SECTION("VectorAddKernelBlockSerial, large block size")
        {
            // Launch the 1-dimensional kernel with a large block size and a single block; this relies on the kernel to
            // check the size of the "problem space" and avoid accessing out-of-bounds data.
            INFO("Test 1D vector block-level serial addition with large block size, using scalar dimensions");
            testVectorAddKernel<TestType>(10, 2, 32, VectorAddKernelBlockSerial{});
        }

        SECTION("VectorAddKernelSkip, small block size")
        {
            // Launch the 1-dimensional kernel with a small block size and a small number of blocks; this relies on the
            // kernel to loop over the "problem space" and do more work per block.
            INFO("Test 1D vector addition with small block size, using scalar dimensions");
            testVectorAddKernelSkip<TestType>(20, 10000, 32, 32, VectorAddKernelSkip{});
        }

        SECTION("VectorAddKernelSkip, large block size")
        {
            // Launch the 1-dimensional kernel with a large block size and a single block; this relies on the kernel to
            // check the size of the "problem space" and avoid accessing out-of-bounds data.
            INFO("Test 1D vector addition with large block size, using scalar dimensions");
            testVectorAddKernelSkip<TestType>(2, 10, 2, 32, VectorAddKernelSkip{});
        }
    }
}
