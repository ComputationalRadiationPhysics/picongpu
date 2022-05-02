/* Copyright 2022 Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch.hpp>

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

template<typename TElem, typename TQueue, typename TExtent>
auto allocAsyncBufIfSupported(TQueue const& queue, TExtent const& extent)
    -> alpaka::Buf<alpaka::Dev<TQueue>, TElem, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>
{
    using Idx = alpaka::Idx<TExtent>;
    if constexpr(alpaka::hasAsyncBufSupport<alpaka::Dev<TQueue>, alpaka::Dim<TExtent>>)
    {
        return alpaka::allocAsyncBuf<TElem, Idx>(queue, extent);
    }
    else
    {
        return alpaka::allocBuf<TElem, Idx>(alpaka::getDev(queue), extent);
    }
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

    using Host = alpaka::DevCpu;
    using HostQueue = alpaka::Queue<Host, alpaka::Blocking>;

    // CPU host
    auto const host = alpaka::getDevByIdx<Host>(0u);
    INFO("using alpaka device: " << alpaka::getName(host))
    HostQueue hostQueue(host);

    // host buffer
    auto h_buffer1 = alpaka::allocBuf<int, Idx>(host, Vec1D{Idx{42}});
    INFO(
        "host buffer allocated at " << alpaka::getPtrNative(h_buffer1) << " with "
                                    << alpaka::getExtentProduct(h_buffer1) << " element(s)")

    // async host buffer
    auto h_buffer2 = allocAsyncBufIfSupported<int>(hostQueue, Vec1D{Idx{42}});
    INFO(
        "second host buffer allocated at " << alpaka::getPtrNative(h_buffer2) << " with "
                                           << alpaka::getExtentProduct(h_buffer2) << " element(s)")

    // host-side memset
    const int value1 = 42;
    const int expected1 = memset_value<int>(value1);
    INFO("host-side memset")
    alpaka::memset(hostQueue, h_buffer1, value1);
    alpaka::wait(hostQueue);
    CHECK(expected1 == *alpaka::getPtrNative(h_buffer1));

    // host-side async memset
    const int value2 = 99;
    const int expected2 = memset_value<int>(value2);
    INFO("host-side async memset")
    alpaka::memset(hostQueue, h_buffer2, value2);
    alpaka::wait(hostQueue);
    CHECK(expected2 == *alpaka::getPtrNative(h_buffer2));

    // host-host copies
    INFO("buffer host-host copies")
    alpaka::memcpy(hostQueue, h_buffer2, h_buffer1);
    alpaka::wait(hostQueue);
    CHECK(expected1 == *alpaka::getPtrNative(h_buffer2));
    alpaka::memcpy(hostQueue, h_buffer1, h_buffer2);
    alpaka::wait(hostQueue);
    CHECK(expected1 == *alpaka::getPtrNative(h_buffer1));

    // GPU device
    auto const device = alpaka::getDevByIdx<Device>(0u);
    INFO("using alpaka device: " << alpaka::getName(device))
    DeviceQueue deviceQueue(device);

    // device buffer
    auto d_buffer1 = alpaka::allocBuf<int, Idx>(device, Vec1D{Idx{42}});
    INFO(
        "device buffer allocated at " << alpaka::getPtrNative(d_buffer1) << " with "
                                      << alpaka::getExtentProduct(d_buffer1) << " element(s)")

    // async or second sync device buffer
    auto d_buffer2 = allocAsyncBufIfSupported<int>(deviceQueue, Vec1D{Idx{42}});
    INFO(
        "second device buffer allocated at " << alpaka::getPtrNative(d_buffer2) << " with "
                                             << alpaka::getExtentProduct(d_buffer2) << " element(s)")

    // host-device copies
    INFO("host-device copies")
    alpaka::memcpy(deviceQueue, d_buffer1, h_buffer1);
    alpaka::memcpy(deviceQueue, d_buffer2, h_buffer2);

    // device-device copies
    INFO("device-device copies")
    alpaka::memcpy(deviceQueue, d_buffer1, d_buffer2);
    alpaka::memcpy(deviceQueue, d_buffer2, d_buffer1);

    // device-side memset
    INFO("device-side memset")
    alpaka::memset(deviceQueue, d_buffer1, value1);
    alpaka::memset(deviceQueue, d_buffer2, value2);

    // device-host copies
    INFO("device-host copies")
    alpaka::memcpy(deviceQueue, h_buffer1, d_buffer1);
    alpaka::memcpy(deviceQueue, h_buffer2, d_buffer2);

    alpaka::wait(deviceQueue);
    CHECK(expected1 == *alpaka::getPtrNative(h_buffer1));
    CHECK(expected2 == *alpaka::getPtrNative(h_buffer2));
}
