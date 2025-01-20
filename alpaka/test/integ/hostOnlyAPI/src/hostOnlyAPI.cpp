/* Copyright 2022 Andrea Bocci, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_message.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>

// fill an trivial type with std::memset
template<typename T>
constexpr auto memset_value(int c) -> T
{
    T t;
    std::memset(&t, c, sizeof(T));
    return t;
}

// 0- and 1- dimensional space
using Idx = std::size_t;
using Dim1D = alpaka::DimInt<1u>;
using Vec1D = alpaka::Vec<Dim1D, Idx>;

// enabled accelerators with 1-dimensional kernel space
using TestAccs = alpaka::test::EnabledAccs<Dim1D, Idx>;

TEMPLATE_LIST_TEST_CASE("hostOnlyAPI", "[hostOnlyAPI]", TestAccs)
{
    using Device = alpaka::Dev<TestType>;
    using DeviceQueue = alpaka::Queue<Device, alpaka::NonBlocking>;

    auto const platformAcc = alpaka::Platform<TestType>{};

    using Host = alpaka::DevCpu;
    using HostQueue = alpaka::Queue<Host, alpaka::Blocking>;

    // CPU host
    auto const platformHost = alpaka::PlatformCpu{};
    auto const host = alpaka::getDevByIdx(platformHost, 0);
    INFO("using alpaka device: " << alpaka::getName(host));
    HostQueue hostQueue(host);

    // host buffer
    auto h_buffer1 = alpaka::allocMappedBufIfSupported<int, Idx>(host, platformAcc, Vec1D{Idx{42}});
    INFO(
        "host buffer allocated at " << std::data(h_buffer1) << " with " << alpaka::getExtentProduct(h_buffer1)
                                    << " element(s)");

    // async host buffer
    auto h_buffer2 = alpaka::allocAsyncBufIfSupported<int, Idx>(hostQueue, Vec1D{Idx{42}});
    INFO(
        "second host buffer allocated at " << std::data(h_buffer2) << " with " << alpaka::getExtentProduct(h_buffer2)
                                           << " element(s)");

    // host-side memset
    int const value1 = 42;
    int const expected1 = memset_value<int>(value1);
    INFO("host-side memset");
    alpaka::memset(hostQueue, h_buffer1, value1);
    alpaka::wait(hostQueue);
    CHECK(expected1 == *std::data(h_buffer1));

    // host-side async memset
    int const value2 = 99;
    int const expected2 = memset_value<int>(value2);
    INFO("host-side async memset");
    alpaka::memset(hostQueue, h_buffer2, value2);
    alpaka::wait(hostQueue);
    CHECK(expected2 == *std::data(h_buffer2));

    // host-host copies
    INFO("buffer host-host copies");
    alpaka::memcpy(hostQueue, h_buffer2, h_buffer1);
    alpaka::wait(hostQueue);
    CHECK(expected1 == *std::data(h_buffer2));
    alpaka::memcpy(hostQueue, h_buffer1, h_buffer2);
    alpaka::wait(hostQueue);
    CHECK(expected1 == *std::data(h_buffer1));

    // GPU device
    auto const device = alpaka::getDevByIdx(platformAcc, 0);
    INFO("using alpaka device: " << alpaka::getName(device));
    DeviceQueue deviceQueue(device);

    // device buffer
    auto d_buffer1 = alpaka::allocBuf<int, Idx>(device, Vec1D{Idx{42}});
    INFO(
        "device buffer allocated at " << std::data(d_buffer1) << " with " << alpaka::getExtentProduct(d_buffer1)
                                      << " element(s)");

    // async or second sync device buffer
    auto d_buffer2 = alpaka::allocAsyncBufIfSupported<int, Idx>(deviceQueue, Vec1D{Idx{42}});
    INFO(
        "second device buffer allocated at " << std::data(d_buffer2) << " with " << alpaka::getExtentProduct(d_buffer2)
                                             << " element(s)");

    // host-device copies
    INFO("host-device copies");
    alpaka::memcpy(deviceQueue, d_buffer1, h_buffer1);
    alpaka::memcpy(deviceQueue, d_buffer2, h_buffer2);

    // device-device copies
    INFO("device-device copies");
    alpaka::memcpy(deviceQueue, d_buffer1, d_buffer2);
    alpaka::memcpy(deviceQueue, d_buffer2, d_buffer1);

    // device-side memset
    INFO("device-side memset");
    alpaka::memset(deviceQueue, d_buffer1, value1);
    alpaka::memset(deviceQueue, d_buffer2, value2);

    // device-host copies
    INFO("device-host copies");
    alpaka::memcpy(deviceQueue, h_buffer1, d_buffer1);
    alpaka::memcpy(deviceQueue, h_buffer2, d_buffer2);

    alpaka::wait(deviceQueue);
    CHECK(expected1 == *std::data(h_buffer1));
    CHECK(expected2 == *std::data(h_buffer2));
}
