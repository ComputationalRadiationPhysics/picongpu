/* Copyright 2024 Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <pmacc/boost_workaround.hpp>

#include <pmacc/test/PMaccFixture.hpp>

// STL
#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/memory/buffers/DeviceBuffer.hpp>
#include <pmacc/memory/buffers/HostBuffer.hpp>
#include <pmacc/meta/conversion/MakeSeq.hpp>

#include <cstdint>
#include <iostream>
#include <string>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <picongpu/param/precision.param>
#include <picongpu/plugins/radiation/windowFunctions.hpp>


//! Helper to setup the PMacc environment
using TestFixture = pmacc::test::PMaccFixture<TEST_DIM>;
static TestFixture fixture;

using namespace picongpu;
using namespace pmacc;

constexpr uint32_t numValues = 1024;
constexpr uint32_t elemPerBlock = 256;

/** check if floating point result is equal
 *
 * Allows an error of one epsilon.
 * @return true if equal, else false
 */
template<typename T>
static bool isApproxEqual(T const& a, T const& b)
{
    return a == Catch::Approx(b).margin(std::numeric_limits<T>::epsilon());
}

/** Test radiation window function
 *
 * Compares the on host and on device computed result.
 *
 * @tparam T_WindowFunction radiation window function
 */
template<typename T_WindowFunction = boost::mpl::_1>
struct TestWindowFunction
{
    /** Validates the shape
     *
     * @param inCellPositionBuffer Buffer with positions, each value must be in rage [0.0;1.0).
     * @param inLxBuffer Buffer with length of the simulated area [0.0;Inf)
     */
    template<typename T_PosBuffer, typename T_LxBuffer>
    void operator()(T_PosBuffer& inCellPositionBuffer, T_LxBuffer& inLxBuffer)
    {
        std::cout << "Test window function" << typeid(T_WindowFunction).name() << std::endl;
        ::pmacc::DeviceBuffer<float_X, 1u> deviceInCellPositionBuffer(numValues);
        ::pmacc::DeviceBuffer<float_X, 1u> deviceLxBuffer(numValues);

        ::pmacc::HostBuffer<float_X, 1u> resultHost(numValues);
        ::pmacc::DeviceBuffer<float_X, 1u> resultDevice(numValues);

        deviceInCellPositionBuffer.copyFrom(inCellPositionBuffer);
        deviceLxBuffer.copyFrom(inLxBuffer);
        resultDevice.setValue(0.0_X);

        auto shapeTestKernel
            = [this] DEVICEONLY(auto const& worker, auto const& positions, auto const& lx, auto result)
        {
            auto blockIdx = worker.blockDomIdxND().x();

            auto forEach = lockstep::makeForEach<elemPerBlock>(worker);

            forEach(
                [&](uint32_t const idx)
                {
                    auto valueIdx = blockIdx * elemPerBlock + idx;

                    if(valueIdx < numValues)
                    {
                        auto windowFunc = T_WindowFunction{};

                        result[valueIdx] = windowFunc(positions[valueIdx], lx[valueIdx]);
                    }
                });
        };

        auto numBlocks = (numValues + elemPerBlock - 1) / elemPerBlock;
        PMACC_LOCKSTEP_KERNEL(shapeTestKernel)
            .template config<elemPerBlock>(numBlocks)(
                deviceInCellPositionBuffer.getDataBox(),
                deviceLxBuffer.getDataBox(),
                resultDevice.getDataBox());

        resultHost.copyFrom(resultDevice);

        auto res = resultHost.getDataBox();
        for(uint32_t i = 0u; i < numValues; ++i)
        {
            auto pos = inCellPositionBuffer.getDataBox()[i];
            auto lx = inLxBuffer.getDataBox()[i];
            auto hostVal = T_WindowFunction{}(pos, lx);
            auto isCorrect = isApproxEqual(hostVal, res[i]);
            if(!isCorrect)
                std::cerr << "pos=" << pos << " lx=" << lx << " result=" << res[i] << std::endl;
            REQUIRE(isCorrect);
        }
    }
};

TEST_CASE("unit::windowFunction", "[radiation window test]")
{
    ::pmacc::HostBuffer<float_X, 1u> inCellPositionBuffer(numValues);
    ::pmacc::HostBuffer<float_X, 1u> lxBuffer(numValues);

    std::mt19937 mt(42.0);
    std::uniform_real_distribution<> dist(0.0, 1.0);
    std::uniform_real_distribution<> distLx(0.0, 9999999.0);

    auto posBox = inCellPositionBuffer.getDataBox();
    auto lxBox = lxBuffer.getDataBox();
    // provide random in cell positions
    for(uint32_t i = 0u; i < numValues; ++i)
    {
        posBox[i] = dist(mt);
        lxBox[i] = distLx(mt);
    }


    // check on support assignment shape
    using WindowFunctions = pmacc::MakeSeq_t<
        plugins::radiation::radWindowFunctionTriangle::RadWindowFunction,
        plugins::radiation::radWindowFunctionHamming::RadWindowFunction,
        plugins::radiation::radWindowFunctionTriplett::RadWindowFunction,
        plugins::radiation::radWindowFunctionGauss::RadWindowFunction,
        plugins::radiation::radWindowFunctionNone::RadWindowFunction>;

    meta::ForEach<WindowFunctions, TestWindowFunction<>>{}(inCellPositionBuffer, lxBuffer);
}
