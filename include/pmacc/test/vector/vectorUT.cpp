/* Copyright 2024 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <pmacc/boost_workaround.hpp>

#include <pmacc/Environment.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/test/PMaccFixture.hpp>
#include <pmacc/verify.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>

#include <catch2/catch_test_macros.hpp>

/** @file
 *
 *  This file is testing vector functionality
 */

using MyPMaccFixture = pmacc::test::PMaccFixture<TEST_DIM>;
static MyPMaccFixture fixture;

TEST_CASE("vector constructor generator", "[vector]")
{
    using namespace pmacc;
    uint32_t const numElements = 2u;
    using VecType = pmacc::math::Vector<uint32_t, TEST_DIM>;

    auto hostDeviceBuffer = HostDeviceBuffer<VecType, DIM1>(DataSpace<DIM1>{numElements});
    using DeviceBuf = DeviceBuffer<uint32_t, DIM1>;

    hostDeviceBuffer.getDeviceBuffer().setValue(VecType::create(42));
    auto const testKernel = [] ALPAKA_FN_ACC(auto const& acc, VecType* data)
    {
        // constexpr lambda generator
        constexpr auto vec = pmacc::math::Vector<uint32_t, TEST_DIM>([](uint32_t const i) constexpr { return i; });
        data[0] = vec;
        // non constexpr lambda generator
        data[1] = pmacc::math::Vector<uint32_t, TEST_DIM>([](uint32_t const i) { return i * 2u; });
    };
    PMACC_KERNEL(testKernel)(1, 1)(hostDeviceBuffer.getDeviceBuffer().data());
    hostDeviceBuffer.deviceToHost();

    REQUIRE(
        hostDeviceBuffer.getHostBuffer().data()[0]
        == pmacc::math::Vector<uint32_t, 3u>(0u, 1u, 2u).shrink<TEST_DIM>());
    REQUIRE(
        hostDeviceBuffer.getHostBuffer().data()[1]
        == pmacc::math::Vector<uint32_t, 3u>(0u, 2u, 4u).shrink<TEST_DIM>());
}
