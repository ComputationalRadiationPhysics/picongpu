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
#include <pmacc/math/ConstVector.hpp>
#include <pmacc/memory/buffers/DeviceBuffer.hpp>
#include <pmacc/memory/buffers/HostBuffer.hpp>
#include <pmacc/meta/conversion/MakeSeq.hpp>

#include <cstdint>
#include <iostream>
#include <string>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <picongpu/param/precision.param>

/* Values must be defined because they are required by the radiation functions.
 * Typically these values coming from simulation_defines but we try to avoid including these in unit tests.
 */
namespace picongpu
{
    constexpr picongpu::float_X SPEED_OF_LIGHT = 1.0;
    constexpr picongpu::float_X CELL_WIDTH = 2.0;
    constexpr picongpu::float_X CELL_HEIGHT = 3.0;
    constexpr picongpu::float_X CELL_DEPTH = 4.0;
    constexpr picongpu::float_X DELTA_T = 5.0;

    PMACC_CONST_VECTOR(picongpu::float_X, TEST_DIM, cellSize, CELL_WIDTH, CELL_HEIGHT, CELL_DEPTH);
} // namespace picongpu

#include <picongpu/plugins/radiation/VectorTypes.hpp>
#include <picongpu/plugins/radiation/radFormFactor.hpp>
#include <picongpu/plugins/radiation/vector.hpp>


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

/** Test radiation form factor function
 *
 * Compares the on host and on device computed result.
 *
 * @tparam T_FormFactorFunctions radiation window function
 */
template<typename T_FormFactorFunctions = boost::mpl::_1>
struct TestFormFactor
{
    /** Validates the shape
     *
     * @param inOmegaBuffer Buffer with omega [0.0;Inf)
     * @param inMacroWeightingBuffer Buffer with makro particle weighting [0.0;Inf)
     */
    template<typename T_OmegaBuffer, typename T_MacroWeightingBUffer>
    void operator()(T_OmegaBuffer& inOmegaBuffer, T_MacroWeightingBUffer& inMacroWeightingBuffer)
    {
        std::cout << "Test form factor function" << typeid(T_FormFactorFunctions).name() << std::endl;
        ::pmacc::DeviceBuffer<float_X, 1u> deviceInOmegaBuffer(numValues);
        ::pmacc::DeviceBuffer<float_X, 1u> deviceInMacroWeightingBuffer(numValues);

        ::pmacc::HostBuffer<float_X, 1u> resultHost(numValues);
        ::pmacc::DeviceBuffer<float_X, 1u> resultDevice(numValues);

        deviceInOmegaBuffer.copyFrom(inOmegaBuffer);
        deviceInMacroWeightingBuffer.copyFrom(inMacroWeightingBuffer);

        resultDevice.setValue(0.0_X);

        auto shapeTestKernel
            = [this] DEVICEONLY(auto const& worker, auto const& omega, auto const& weighting, auto result)
        {
            auto blockIdx = worker.blockDomIdxND().x();

            auto forEach = lockstep::makeForEach<elemPerBlock>(worker);

            forEach(
                [&](uint32_t const idx)
                {
                    auto valueIdx = blockIdx * elemPerBlock + idx;

                    if(valueIdx < numValues)
                    {
                        float_X const omg = omega[valueIdx];
                        auto const windowFunc = T_FormFactorFunctions{omg, {1.0, 1.0, 1.0}};

                        result[valueIdx] = windowFunc(weighting[valueIdx]);
                    }
                });
        };

        auto numBlocks = (numValues + elemPerBlock - 1) / elemPerBlock;
        PMACC_LOCKSTEP_KERNEL(shapeTestKernel)
            .template config<elemPerBlock>(numBlocks)(
                deviceInOmegaBuffer.getDataBox(),
                deviceInMacroWeightingBuffer.getDataBox(),
                resultDevice.getDataBox());

        resultHost.copyFrom(resultDevice);

        auto res = resultHost.getDataBox();
        for(uint32_t i = 0u; i < numValues; ++i)
        {
            auto omega = inOmegaBuffer.getDataBox()[i];
            auto weighting = inMacroWeightingBuffer.getDataBox()[i];
            auto hostVal = T_FormFactorFunctions(omega, {1.0, 1.0, 1.0})(weighting);
            auto isCorrect = isApproxEqual(hostVal, res[i]);
            if(!isCorrect)
                std::cerr << "omega=" << omega << " weighting=" << weighting << " result=" << res[i] << std::endl;
            REQUIRE(isCorrect);
        }
    }
};

TEST_CASE("unit::windowFormFactor", "[radiation window formfactor test]")
{
    ::pmacc::HostBuffer<float_X, 1u> inOmegaBuffer(numValues);
    ::pmacc::HostBuffer<float_X, 1u> inMacroWeightingBuffer(numValues);

    std::mt19937 mt(42.0);
    std::uniform_real_distribution<> dist(0.0, 999999.0);

    auto omegaBox = inOmegaBuffer.getDataBox();
    auto weightingBox = inMacroWeightingBuffer.getDataBox();
    // provide random in cell positions
    for(uint32_t i = 0u; i < numValues; ++i)
    {
        omegaBox[i] = dist(mt);
        weightingBox[i] = dist(mt);
    }

    // check on support assignment shape
    using FormFactorFunctions = pmacc::MakeSeq_t<
        plugins::radiation::radFormFactor_CIC_3D::RadFormFactor,
        plugins::radiation::radFormFactor_TSC_3D::RadFormFactor,
        plugins::radiation::radFormFactor_PCS_3D::RadFormFactor,
        plugins::radiation::radFormFactor_CIC_1Dy::RadFormFactor,
        plugins::radiation::radFormFactor_Gauss_spherical::RadFormFactor,
        plugins::radiation::radFormFactor_Gauss_cell::RadFormFactor,
        plugins::radiation::radFormFactor_incoherent::RadFormFactor,
        plugins::radiation::radFormFactor_coherent::RadFormFactor>;

    meta::ForEach<FormFactorFunctions, TestFormFactor<>>{}(inOmegaBuffer, inMacroWeightingBuffer);
}
