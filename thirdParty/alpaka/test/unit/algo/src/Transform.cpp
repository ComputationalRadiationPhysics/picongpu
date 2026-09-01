/* Copyright 2025 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <random>

// A functor that takes a single argument of type float and returns a value of type float
struct Duplicate
{
    template<typename T>
    ALPAKA_FN_HOST_ACC T operator()(T value) const
    {
        return value * 2;
    }
};

// A functor that takes two arguments (an accelerator object and a float) and returns a value of type float
struct Sin
{
    template<typename TAcc, typename T>
    ALPAKA_FN_HOST_ACC T operator()(TAcc const& acc, T value) const
    {
        return alpaka::math::sin(acc, value);
    }
};

using TestAccs1D = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, uint32_t>;

TEMPLATE_LIST_TEST_CASE("transform_pointers", "[algo]", TestAccs1D)
{
    using Acc = TestType;
    using Tag = alpaka::AccToTag<Acc>;
    using Platform = alpaka::Platform<Acc>;
    using Device = alpaka::Dev<Acc>;
    using Queue = alpaka::Queue<Device, alpaka::NonBlocking>;

    using Idx = alpaka::Idx<Acc>;
    // using Dim = alpaka::Dim<Acc>;
    // using Vec = alpaka::Vec<Dim, Idx>;

    const alpaka::PlatformCpu host_platform{};
    auto const host = alpaka::getDevByIdx(host_platform, 0u);

    const Platform platform{};
    const Device device = alpaka::getDevByIdx(platform, 0);
    Queue queue{device};

    // Random number generator with a gaussian distribution.
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0.f, 1.f};

    Idx size = 1000;

    // Allocate input and output buffers on the host.
    auto sample = alpaka::allocMappedBuf<float, Idx>(host, platform, size);
    auto result = alpaka::allocMappedBuf<float, Idx>(host, platform, size);

    // Fill the input buffer with random numbers.
    for(Idx i = 0; i < size; ++i)
    {
        sample[i] = dist(rand);
    }

    // Allocate input and output buffers on the device.
    auto sample_d = alpaka::allocAsyncBufIfSupported<float, Idx>(queue, size);
    auto result_d = alpaka::allocAsyncBufIfSupported<float, Idx>(queue, size);

    // Copy the input random numbers to the device.
    alpaka::memcpy(queue, sample_d, sample);

    // Apply the transform algorithm on the device.
    alpaka::transform<Tag>(queue, result_d.data(), result_d.data() + size, Duplicate{}, sample_d.data());

    // Copy the result to the host.
    alpaka::memcpy(queue, result, result_d);

    // Wait for all asynchronous operations to complete.
    alpaka::wait(queue);

    // Check the correctness of the results.
    for(Idx i = 0; i < size; ++i)
    {
        float expected = Duplicate{}(sample[i]);
        REQUIRE_THAT(result[i], Catch::Matchers::WithinULP(expected, 4));
    }
}

TEMPLATE_LIST_TEST_CASE("transform_buffer", "[algo]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Tag = alpaka::AccToTag<Acc>;
    using Platform = alpaka::Platform<Acc>;
    using Device = alpaka::Dev<Acc>;
    using Queue = alpaka::test::DefaultQueue<Device>;

    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;
    using Vec = alpaka::Vec<Dim, Idx>;

    const alpaka::PlatformCpu host_platform{};
    auto const host = alpaka::getDevByIdx(host_platform, 0u);

    const Platform platform{};
    const Device device = alpaka::getDevByIdx(platform, 0);
    Queue queue{device};

    // Random number generator with a gaussian distribution.
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0.f, 1.f};

    Vec size = alpaka::test::extentBuf<Dim, Idx>;

    // Allocate input and output buffers on the host.
    auto sample = alpaka::allocMappedBuf<float, Idx>(host, platform, size);
    auto result = alpaka::allocMappedBuf<float, Idx>(host, platform, size);

    // Fill the input buffer with random numbers.
    for(Idx i = 0; i < size.prod(); ++i)
    {
        sample.data()[i] = dist(rand);
    }

    // Allocate input and output buffers on the device.
    auto sample_d = alpaka::allocAsyncBufIfSupported<float, Idx>(queue, size);
    auto result_d = alpaka::allocAsyncBufIfSupported<float, Idx>(queue, size);

    // Copy the input random numbers to the device.
    alpaka::memcpy(queue, sample_d, sample);

    // Apply the transform algorithm on the device.
    alpaka::transform<Tag>(queue, result_d, Sin{}, sample_d);

    // Copy the result to the host.
    alpaka::memcpy(queue, result, result_d);

    // Wait for all asynchronous operations to complete.
    alpaka::wait(queue);

    // Check the correctness of the results.
    for(Idx i = 0; i < size.prod(); ++i)
    {
        float expected = std::sin(sample.data()[i]);
        REQUIRE_THAT(result.data()[i], Catch::Matchers::WithinULP(expected, 4));
    }
}
