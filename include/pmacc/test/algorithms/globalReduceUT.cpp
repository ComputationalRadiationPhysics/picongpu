/* Copyright 2026 NopeCharon
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
#include <pmacc/algorithms/GlobalReduce.hpp>
#include <pmacc/math/operation/Add.hpp>
#include <pmacc/math/operation/Max.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/mpi/reduceMethods/AllReduce.hpp>
#include <pmacc/test/PMaccFixture.hpp>

#include <cstdint>

#include <catch2/catch_test_macros.hpp>

/** @file
 *
 *  This file is testing the reduction of device memory over all MPI ranks.
 */

using MyPMaccFixture = pmacc::test::PMaccFixture<TEST_DIM>;
static MyPMaccFixture fixture;

constexpr uint32_t numElements = 4096u;
//! sum of the values 0, 1, ..., numElements - 1
constexpr uint32_t expectedSum = numElements * (numElements - 1u) / 2u;

//! fill the device buffer with the values 0, 1, ..., numElements - 1
template<typename T_HostDeviceBuffer>
inline void fillIota(T_HostDeviceBuffer& buffer)
{
    auto hostBox = buffer.getHostBuffer().getDataBox();
    for(uint32_t i = 0u; i < numElements; ++i)
        hostBox[i] = i;
    buffer.hostToDevice();
}

TEST_CASE("algorithms::GlobalReduce default participation", "[GlobalReduce]")
{
    using namespace pmacc;
    HostDeviceBuffer<uint32_t, DIM1> buffer(DataSpace<DIM1>{numElements});
    fillIota(buffer);

    // participate() is never called, all MPI ranks must take part in the reduction by default
    algorithms::GlobalReduce globalReduce(1024);
    auto deviceBox = buffer.getDeviceBuffer().getDataBox();

    uint32_t const sum = globalReduce(math::operation::Add(), deviceBox, numElements);
    uint32_t const max = globalReduce(math::operation::Max(), deviceBox, numElements);

    REQUIRE(globalReduce.hasResult(mpi::reduceMethods::AllReduce()));
    CHECK(sum == expectedSum);
    CHECK(max == numElements - 1u);
}

TEST_CASE("algorithms::GlobalReduce explicit participation", "[GlobalReduce]")
{
    using namespace pmacc;
    HostDeviceBuffer<uint32_t, DIM1> buffer(DataSpace<DIM1>{numElements});
    fillIota(buffer);

    algorithms::GlobalReduce globalReduce(1024);
    auto deviceBox = buffer.getDeviceBuffer().getDataBox();

    globalReduce.participate(true);
    CHECK(globalReduce(math::operation::Add(), deviceBox, numElements) == expectedSum);

    // a rank which is not participating gets a default constructed result
    globalReduce.participate(false);
    CHECK(globalReduce(math::operation::Add(), deviceBox, numElements) == 0u);
}
