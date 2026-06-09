/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2024-2025 Rene Widera, Alexander Debus
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
#include <pmacc/algorithms/math.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/math/ConstVector.hpp>
#include <pmacc/memory/buffers/DeviceBuffer.hpp>
#include <pmacc/memory/buffers/HostBuffer.hpp>
#include <pmacc/meta/conversion/MakeSeq.hpp>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <typeinfo>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <picongpu/fields/background/templates/twtstight/TWTSTight.hpp>
#include <picongpu/param/precision.param>

using namespace picongpu;
using namespace pmacc;

//! Helper to setup the PMacc environment
static pmacc::test::PMaccFixture<simDim> pmaccFixture;

/** check if floating point result is equal
 *
 * Allows an error of one epsilon.
 * @return true if equal, else false
 */
template<typename T>
static bool isApproxEqual(T const& a, T const& b, T const& epsilon)
{
    T const epsilonScaled = epsilon * math::max(math::abs(a), math::abs(b));
    return a == Catch::Approx(b).margin(epsilonScaled);
}

template<typename T>
static bool isApproxEqual(T const& a, T const& b)
{
    T const epsilon = std::numeric_limits<T>::epsilon() * math::max(math::abs(a), math::abs(b));
    return a == Catch::Approx(b).margin(epsilon);
}

template<uint32_t T_numThreadsPerBlock>
struct GenerateEvals
{
    templates::twtstight::EField const testEfield;
    templates::twtstight::BField const testBfield;
    using float_T = templates::twtstight::float_T;

    HINLINE GenerateEvals()
        : testEfield(0.0, 800.0e-9, 30.0e-15, 2.5e-6, 5. * (PI / 180.), 1.0, 0.0, false, 0.0, 30. * (PI / 180.))
        , testBfield(0.0, 800.0e-9, 30.0e-15, 2.5e-6, 5. * (PI / 180.), 1.0, 0.0, false, 0.0, 30. * (PI / 180.))
    {
    }

    template<class T_Box, typename T_Worker>
    HDINLINE void operator()(
        T_Worker const& worker,
        uint32_t const numValues,
        uint32_t const numValuesPerThread,
        float3_64 const pos,
        float_64 const time,
        T_Box result) const
    {
        using namespace ::pmacc;
        uint32_t const blockIdx = worker.blockDomIdxND().x();
        auto forEach = lockstep::makeForEach<T_numThreadsPerBlock>(worker);

        forEach(
            [&](uint32_t const idx)
            {
                auto valueIdx = blockIdx * T_numThreadsPerBlock + idx;
                if(valueIdx < numValues)
                {
                    result(0u + valueIdx) = testEfield.calcTWTSFieldX(pos, time);
                    result(1u + valueIdx) = testEfield.calcTWTSFieldY(pos, time);
                    result(2u + valueIdx) = testEfield.calcTWTSFieldZ(pos, time);
                    result(3u + valueIdx) = testBfield.calcTWTSFieldX(pos, time);
                    result(4u + valueIdx) = testBfield.calcTWTSFieldY(pos, time);
                    result(5u + valueIdx) = testBfield.calcTWTSFieldZ(pos, time);
                }
            });
    }
};

/** Test TWTSTight laser functions
 *
 * Compares the on host and on device computed result to analytical results.
 *
 */
struct twtsTightNumberTest
{
    void operator()()
    {
        using namespace ::pmacc;
        templates::twtstight::EField const testEfield = templates::twtstight::EField(
            0.0,
            800.0e-9,
            30.0e-15,
            2.5e-6,
            5. * (PI / 180.),
            1.0,
            0.0,
            false,
            0.0,
            30. * (PI / 180.));
        templates::twtstight::BField const testBfield = templates::twtstight::BField(
            0.0,
            800.0e-9,
            30.0e-15,
            2.5e-6,
            5. * (PI / 180.),
            1.0,
            0.0,
            false,
            0.0,
            30. * (PI / 180.));
        using float_T = templates::twtstight::float_T;
        using float3_T = ::pmacc::math::Vector<float_T, 3u>;

        constexpr uint32_t numBlocks = 1;
        constexpr uint32_t numThreadsPerBlock = 1;
        constexpr uint32_t numThreads = numBlocks * numThreadsPerBlock;
        constexpr uint32_t numValuesPerThread = 6;
        constexpr uint32_t numValues = numThreads * numValuesPerThread;
        float3_64 const pos = float3_64{1.0e-6, 1.0e-6, 100.0 * 1.5e-6};
        float_64 const time = float_64(1.0e-15);

        HostBuffer<float_T, 1u> resultHost(numValues);
        DeviceBuffer<float_T, 1u> resultDevice(numValues);
        resultDevice.setValue(float_T(0.0));

        PMACC_LOCKSTEP_KERNEL(GenerateEvals<numThreadsPerBlock>{})
            .template config<numThreadsPerBlock>(
                numBlocks)(numValues, numValuesPerThread, pos, time, resultDevice.getDataBox());

        resultHost.copyFrom(resultDevice);

        auto res = resultHost.getDataBox();
        auto hostEfield = float3_T(
            testEfield.calcTWTSFieldX(pos, time),
            testEfield.calcTWTSFieldY(pos, time),
            testEfield.calcTWTSFieldZ(pos, time));
        auto hostBfield = float3_T(
            testBfield.calcTWTSFieldX(pos, time),
            testBfield.calcTWTSFieldY(pos, time),
            testBfield.calcTWTSFieldZ(pos, time));

// This combination of compilers has a bug that is triggered by Catch2 internally suppressing warnings.
// See https://github.com/ComputationalRadiationPhysics/picongpu/pull/5174#issuecomment-2467890326
#if (__GNUC__ != 11 || __CUDACC_VER_MAJOR__ != 11)
        const float3_64 refEfield = float3_64(0.18329124052693974, -0.009402050968104002, 0.1054028749666347);
        float3_64 const refBfield = float3_64(3.5299706879027803e-10, 5.334111474127282e-11, -6.090365721598194e-10);
        float3_T const refEfieldT = precisionCast<float_T>(refEfield);
        float3_T const refBfieldT = precisionCast<float_T>(refBfield);
        /* epsilon to compare to Mathematica implementation.
         * Note: Reduction of epsilon would require replacing complex-valued bessel function support in
         * PMacc with boost library calls that also work on device. */
        float_T const epsilonAlgebra = std::is_same<float_T, float_64>::value ? float_T(5.0e-13) : float_T(5.0e-5);
        /* Epsilon to compare host implementation to device implementation */
        float_T const epsilonHostDevice = std::is_same<float_T, float_64>::value ? float_T(5.0e-15) : float_T(5.0e-7);
        for(uint32_t i = 0; i < 3; i++)
        {
            CHECK(isApproxEqual(refEfieldT[i], res[i], epsilonAlgebra));
            CHECK(isApproxEqual(hostEfield[i], res[i], epsilonHostDevice));
        }
        for(uint32_t i = 0; i < 3; i++)
        {
            CHECK(isApproxEqual(refBfieldT[i], res[i + 3], epsilonAlgebra));
            CHECK(isApproxEqual(hostBfield[i], res[i + 3], epsilonHostDevice));
        }
#endif
    }
};

TEST_CASE("unit::TWTSTight", "[TWTSTight laser math test]")
{
    twtsTightNumberTest()();
}
