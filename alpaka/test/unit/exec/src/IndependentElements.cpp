/* Copyright 2024 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include "WorkDiv.hpp"
#include "alpaka/acc/Traits.hpp"
#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/exec/UniformElements.hpp"
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

/* Add the group id to the value of each element in the group.
 * Each group is composed by the elements first[group]..first[group+1]-1 .
 */
struct IndependentWorkKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        T const* __restrict__ in,
        T* __restrict__ out,
        alpaka::Idx<TAcc> const* __restrict__ indices,
        alpaka::Idx<TAcc> groups) const
    {
        using Idx = alpaka::Idx<TAcc>;

        for(auto group : alpaka::independentGroups(acc, groups))
        {
            Idx first = indices[group];
            Idx last = indices[group + 1];
            Idx size = last - first;
            for(auto index : alpaka::independentGroupElements(acc, size))
            {
                out[first + index] = in[first + index] + static_cast<float>(group);
            }
        }
    }
};

/* Test the IndependentWorkKernel kernel on all devices
 */
template<typename TAcc, typename TKernel>
void testIndependentWorkKernel(
    alpaka::Idx<TAcc> groups,
    alpaka::Idx<TAcc> grid_size,
    alpaka::Idx<TAcc> block_size,
    TKernel kernel)
{
    using Acc = TAcc;
    using Idx = alpaka::Idx<Acc>;
    using Platform = alpaka::Platform<Acc>;
    using Device = alpaka::Dev<Platform>;
    using Queue = alpaka::Queue<Device, alpaka::NonBlocking>;

    // Initialise the accelerator platform.
    Platform platform{};

    // Random number generator with a gaussian distribution.
    std::random_device rd{};
    std::default_random_engine engine{rd()};

    // Uniform distribution.
    std::uniform_int_distribution<Idx> random_size{100, 201};

    // Gaussian distribution.
    std::normal_distribution<float> dist{0.f, 1.f};

    // Build the groups.
    auto indices_h = alpaka::allocMappedBuf<Idx, Idx>(host, platform, groups + 1);
    indices_h[0] = 0;
    for(Idx i = 0; i < groups; ++i)
    {
        // Group "i" has "size" elements.
        auto size = random_size(engine);
        indices_h[i + 1] = indices_h[i] + size;
    }

    // Tolerance.
    constexpr float epsilon = 0.000001f;

    // Buffer size.
    const Idx size = indices_h[groups];

    // Allocate the input and output host buffer in pinned memory accessible by the Platform devices.
    auto in_h = alpaka::allocMappedBuf<float, Idx>(host, platform, size);
    auto out_h = alpaka::allocMappedBuf<float, Idx>(host, platform, size);

    // Fill the input buffers with random data, and the output buffer with zeros.
    for(Idx i = 0; i < size; ++i)
    {
        in_h[i] = dist(engine);
        out_h[i] = 0;
    }

    // Run the test on each device.
    for(auto const& device : alpaka::getDevs(platform))
    {
        /* clang-format off */
        INFO("Test IndependentWorkKernel on " << alpaka::getName(device) << " over " << size << " elements in "
                                              << groups << " independent groups with " << grid_size << " blocks of "
                                              << block_size << " elements");
        /* clang-format on */
        auto queue = Queue(device);

        // Allocate input and output buffers on the device.
        auto indices_d = alpaka::allocAsyncBufIfSupported<Idx, Idx>(queue, groups + 1);
        auto in_d = alpaka::allocAsyncBufIfSupported<float, Idx>(queue, size);
        auto out_d = alpaka::allocAsyncBufIfSupported<float, Idx>(queue, size);

        // Copy the input data to the device; the size is known from the buffer objects.
        alpaka::memcpy(queue, indices_d, indices_h);
        alpaka::memcpy(queue, in_d, in_h);

        // Fill the output buffer with zeros; the size is known from the buffer objects.
        alpaka::memset(queue, out_d, 0);

        // Launch the 1-dimensional kernel with independent work groups.
        auto workdiv = makeWorkDiv<TAcc>(grid_size, block_size);
        alpaka::exec<TAcc>(queue, workdiv, kernel, in_d.data(), out_d.data(), indices_d.data(), groups);

        // Copy the results from the device to the host.
        alpaka::memcpy(queue, out_h, out_d);

        // Wait for all the operations to complete.
        alpaka::wait(queue);

        // Check the results.
        for(Idx g = 0; g < groups; ++g)
        {
            Idx first = indices_h[g];
            Idx last = indices_h[g + 1];
            for(Idx i = first; i < last; ++i)
            {
                float sum = in_h[i] + static_cast<float>(g);
                float delta = std::max(std::fabs(sum) * epsilon, epsilon);
                REQUIRE(out_h[i] < sum + delta);
                REQUIRE(out_h[i] > sum - delta);
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE("IndependentElements", "[exec]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;

    // 1-dimensional kernels.
    if constexpr(Dim::value == 1)
    {
        SECTION("IndependentWorkKernel, small block size")
        {
            // Launch the independent work kernel with a small block size and a small number of blocks; this relies on
            // the kernel to loop over the "problem space" and do more work per block.
            INFO("Test independent work kernel with small block size, using scalar dimensions");
            testIndependentWorkKernel<TestType>(100, 32, 32, IndependentWorkKernel{});
        }

        SECTION("IndependentWorkKernel, large block size")
        {
            // Launch the independent work kernel with a large block size and a single block; this relies on the kernel
            // to check the size of the "problem space" and avoid accessing out-of-bounds data.
            INFO("Test independent work kernel with large block size, using scalar dimensions");
            testIndependentWorkKernel<TestType>(10, 1, 32, IndependentWorkKernel{});
        }

        SECTION("IndependentWorkKernel, many large blocks")
        {
            // Launch the independent work kernel with a large block size and a large number of blocks; this relies on
            // the kernel to check the size of the "problem space" and avoid accessing out-of-bounds data.
            INFO("Test independent work kernel with large block size, using scalar dimensions");
            testIndependentWorkKernel<TestType>(10, 32, 32, IndependentWorkKernel{});
        }
    }
}
