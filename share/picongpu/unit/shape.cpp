/* Copyright 2023 Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
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
#include <picongpu/particles/shapes/CIC.hpp>
#include <picongpu/particles/shapes/NGP.hpp>
#include <picongpu/particles/shapes/PCS.hpp>
#include <picongpu/particles/shapes/PQS.hpp>
#include <picongpu/particles/shapes/TSC.hpp>


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

/** Do not shift the in cell position. */
struct NoPositionShift
{
    template<bool isEven>
    HDINLINE float_X shift(float_X pos)
    {
        return pos;
    }
};

/** Shift the in cell position.
 *
 * Shifting the in cell position before querying the on support shape is required to fulfill the pre conditions of the
 * shape function.
 */
struct PositionShift
{
    template<bool isEven>
    HDINLINE float_X shift(float_X pos)
    {
        const float_X v_pos = pos - 0.5_X;
        int intShift;
        if constexpr(isEven)
        {
            // pos range [-1.0;0.5)
            intShift = v_pos >= float_X{-0.5} ? 0 : -1;
        }
        else
        {
            // pos range [-1.0;0.5)
            intShift = v_pos >= float_X{0.0} ? 1 : 0;
        }
        return v_pos - float_X(intShift) + float_X{0.5};
    }
};


/** Test a shape
 *
 * Evaluate the assignment shape at all grid points based on a random particle position.
 * The sum of shape values must be 1.0.
 *
 * @tparam T_Shape assignment shape type, supports shape::ChargeAssignment and shape::ChargeAssignmentOnSupport
 */
template<typename T_Shape = boost::mpl::_1>
struct TestShape
{
    /** Validates the shape
     *
     * @param inCellPositionBuffer Buffer with positions, each value must be in rage [0.0;1.0).
     * @param posShiftFunctor Functor which shifts the position into a valid range to be passed to the shape.
     */
    template<typename T_PosBuffer, typename T_ShiftFunctor>
    void operator()(T_PosBuffer& inCellPositionBuffer, T_ShiftFunctor posShiftFunctor)
    {
        std::cout << "Test Shape" << typeid(T_Shape).name() << std::endl;
        ::pmacc::DeviceBuffer<float_X, 1u> deviceInCellPositionBuffer(numValues);
        ::pmacc::HostBuffer<float_X, 1u> resultHost(numValues);
        ::pmacc::DeviceBuffer<float_X, 1u> resultDevice(numValues);

        deviceInCellPositionBuffer.copyFrom(inCellPositionBuffer);
        resultDevice.setValue(0.0_X);

        auto shapeTestKernel
            = [this] DEVICEONLY(auto const& worker, auto positionShift, auto const& positions, auto result)
        {
            auto blockIdx = worker.blockDomIdxND().x();

            auto forEach = lockstep::makeForEach<elemPerBlock>(worker);

            forEach(
                [&](uint32_t const idx)
                {
                    auto valueIdx = blockIdx * elemPerBlock + idx;

                    if(valueIdx < numValues)
                    {
                        using Shape = T_Shape;
                        auto shape = Shape{};

                        for(int g = Shape::begin; g <= Shape::end; ++g)
                        {
                            // shift the particle position into a valid range to be used to query the shape
                            auto p = positionShift.template shift<Shape::support % 2 == 0>(positions[valueIdx]);
                            result[valueIdx] += shape(g - p);
                        }
                    }
                });
        };

        auto numBlocks = (numValues + elemPerBlock - 1) / elemPerBlock;
        PMACC_LOCKSTEP_KERNEL(shapeTestKernel)
            .template config<elemPerBlock>(
                numBlocks)(posShiftFunctor, deviceInCellPositionBuffer.getDataBox(), resultDevice.getDataBox());

        resultHost.copyFrom(resultDevice);

        auto res = resultHost.getDataBox();
        for(uint32_t i = 0u; i < numValues; ++i)
        {
            auto isCorrect = isApproxEqual(res[i], 1.0_X);
            if(!isCorrect)
                std::cerr << "pos=" << inCellPositionBuffer.getDataBox()[i] << " sum=" << res[i] << std::endl;
            REQUIRE(isCorrect);
        }
    }
};

TEST_CASE("unit::shape", "[shape test]")
{
    ::pmacc::HostBuffer<float_X, 1u> inCellPositionBuffer(numValues);

    std::mt19937 mt(42.0);
    std::uniform_real_distribution<> dist(0.0, 1.0);

    auto posBox = inCellPositionBuffer.getDataBox();
    // provide random in cell positions
    for(uint32_t i = 0u; i < numValues; ++i)
        posBox[i] = dist(mt);

    // check on support assignment shape
    using OnSupportShapes = pmacc::MakeSeq_t<
        particles::shapes::NGP::ChargeAssignmentOnSupport,
        particles::shapes::CIC::ChargeAssignmentOnSupport,
        particles::shapes::TSC::ChargeAssignmentOnSupport,
        particles::shapes::PQS::ChargeAssignmentOnSupport,
        particles::shapes::PCS::ChargeAssignmentOnSupport>;

    meta::ForEach<OnSupportShapes, TestShape<>>{}(inCellPositionBuffer, PositionShift{});

    // check assignment shape outside of the support
    using NotOnSupportShapes = pmacc::MakeSeq_t<
        particles::shapes::NGP::ChargeAssignment,
        particles::shapes::CIC::ChargeAssignment,
        particles::shapes::TSC::ChargeAssignment,
        particles::shapes::PQS::ChargeAssignment,
        particles::shapes::PCS::ChargeAssignment>;

    meta::ForEach<NotOnSupportShapes, TestShape<>>{}(inCellPositionBuffer, NoPositionShift{});
}
