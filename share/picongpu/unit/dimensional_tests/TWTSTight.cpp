/* Copyright 2024-2026 Rene Widera, Alexander Debus
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
        const float3_64 refEfield = float3_64(0.17315102932506113, -0.008881928245568242, 0.09957167445899125);
        float3_64 const refBfield = float3_64(3.3346823460024014e-10, 5.0390221021188766e-11, -5.753428577396939e-10);
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

/** Regression test for the TWTS envelope zero-cutoff guard
 *
 * The guard must test the same envelope argument (t - nu - xi) that is used by
 * the field calculation. Before the fix it tested a differently oriented plane
 * and therefore spuriously zeroed field values close to the envelope center.
 * ``isOutsideTWTSEnvelope`` is public, so it is probed directly: this is
 * deterministic and has no dependence on the oscillatory field amplitude.
 */
// This combination of compilers has a bug that is triggered by Catch2 internally suppressing warnings.
// See https://github.com/ComputationalRadiationPhysics/picongpu/pull/5174#issuecomment-2467890326
#if (__GNUC__ != 11 || __CUDACC_VER_MAJOR__ != 11)
struct twtsTightEnvelopeGuardTest
{
    /** Walk along z at fixed y and t from the envelope center across its
     *  boundary at ``numSigmas * tauG * cspeed``. Points up to half the
     *  boundary distance inside must not be classified as outside, while a
     *  point two boundary distances away must be outside.
     */
    static void checkEnvelope(float_64 const beta_0, float_64 const phi)
    {
        using float_T = templates::twtstight::float_T;

        templates::twtstight::EField const
            testEfield{0.0, 800.0e-9, 30.0e-15, 2.5e-6, phi, beta_0, 0.0, false, 0.0, 30. * (PI / 180.)};
        auto const& vars = testEfield.basicTWTSHelperVariables;
        float_T const sinPhi = vars[1];
        float_T const cosPhi = vars[2];
        float_T const tanAlpha = vars[4];
        float_T const cspeed = vars[5];
        float_T const tauG = vars[8];

        /* Envelope argument: t - nu - xi
         *   = (cspeed*t - y*(cosPhi + sinPhi*tanAlpha) - z*(sinPhi - cosPhi*tanAlpha)) / cspeed
         */
        float_T const yCoeff = cosPhi + sinPhi * tanAlpha;
        float_T const zCoeff = sinPhi - cosPhi * tanAlpha;
        float_T const boundary = float_T(templates::twtstight::numSigmas) * tauG * cspeed;
        float_T const y = float_T(50.0) * boundary;
        float_T const t = float_T(0.0);
        /* z of the envelope center line for the chosen y and t */
        float_T const zCenter = (cspeed * t - y * yCoeff) / zCoeff;

        INFO("beta_0 = " << beta_0 << ", phi = " << phi);
        {
            float_T const z = zCenter;
            CHECK_FALSE(testEfield.isOutsideTWTSEnvelope(std::array<float_T, 4u>{float_T(0.0), y, z, t}));
        }
        {
            float_T const z = zCenter - float_T(0.5) * boundary / zCoeff;
            CHECK_FALSE(testEfield.isOutsideTWTSEnvelope(std::array<float_T, 4u>{float_T(0.0), y, z, t}));
        }
        {
            float_T const z = zCenter - float_T(2.0) * boundary / zCoeff;
            CHECK(testEfield.isOutsideTWTSEnvelope(std::array<float_T, 4u>{float_T(0.0), y, z, t}));
        }
    }

    /** End-to-end check of the negative interaction angle (phi < 0) sign path.
     *
     * ``defineMinimalCoordinates`` applies ``phiPositive`` (negation of x and
     * z) for phi < 0 and evaluates ``deltaT`` with ``cos(phi) = cos(|phi|)``,
     * while the guard works on helpers built from ``abs(phi)`` and is therefore
     * sign invariant. This check drives ``calcTWTSField*`` through
     * ``defineMinimalCoordinates`` with a negative phi and host coordinates
     * derived from that mapping, asserting a non-zero field inside the envelope
     * and an exactly zero field beyond the zero cutoff.
     */
    static void checkNegativePhi()
    {
        using float_T = templates::twtstight::float_T;

        float_64 const phi = -30. * (PI / 180.);
        templates::twtstight::EField const
            testEfield{0.0, 800.0e-9, 30.0e-15, 2.5e-6, phi, 1.0, 0.0, false, 0.0, 30. * (PI / 180.)};
        auto const& vars = testEfield.basicTWTSHelperVariables;
        float_T const tanAlpha = vars[4];
        float_64 const phiPositive = testEfield.phiPositive;
        float_64 const unitLength = testEfield.unit_length;
        float_64 const unitTime = testEfield.dt;

        /* For beta_0 = 1 the guard argument reduces to t - y - z*tanAlpha, so
         * the envelope center line is y = t - z*tanAlpha. The chosen reduced
         * time is below one period, deltaT = wavelength / c / (1 - beta_0*cos(phi)),
         * hence ``numberOfPeriods`` is zero and ``defineMinimalCoordinates``
         * maps pos.y()/time directly while negating x and z via phiPositive.
         * Invert that mapping to obtain the host position and time.
         */
        float_64 const insideZ = 9000.0;
        float_64 const insideT = 100.0;
        float_64 const insideY = insideT - insideZ * tanAlpha;
        float3_64 const posInside = float3_64{0.0, insideY * unitLength, phiPositive * insideZ * unitLength};
        float_64 const timeInside = insideT * unitTime;

        INFO("phi = " << phi);
        CHECK(testEfield.calcTWTSFieldX(posInside, timeInside) != float_T(0.0));
        CHECK(testEfield.calcTWTSFieldY(posInside, timeInside) != float_T(0.0));
        CHECK(testEfield.calcTWTSFieldZ(posInside, timeInside) != float_T(0.0));

        /* |t - y - z*tanAlpha| = 20000*tanAlpha exceeds the cutoff
         * numSigmas*tauG*cspeed, so all field components must be zero. */
        float3_64 const posOutside = float3_64{0.0, 0.0, phiPositive * 20000.0 * unitLength};
        CHECK(testEfield.calcTWTSFieldX(posOutside, 0.0) == float_T(0.0));
        CHECK(testEfield.calcTWTSFieldY(posOutside, 0.0) == float_T(0.0));
        CHECK(testEfield.calcTWTSFieldZ(posOutside, 0.0) == float_T(0.0));
    }

    void operator()() const
    {
        checkEnvelope(1.0, 30. * (PI / 180.));
        checkEnvelope(1.0, 90. * (PI / 180.));
        checkEnvelope(0.9, 30. * (PI / 180.));
        checkEnvelope(0.9, 90. * (PI / 180.));
        checkNegativePhi();
    }
};
#endif

TEST_CASE("unit::TWTSTight", "[TWTSTight laser math test]")
{
    twtsTightNumberTest()();
}

TEST_CASE("unit::TWTSTightEnvelopeGuard", "[TWTSTight laser envelope guard test]")
{
#if (__GNUC__ != 11 || __CUDACC_VER_MAJOR__ != 11)
    twtsTightEnvelopeGuardTest{}();
#endif
}
