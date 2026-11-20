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

#include <array>
#include <cmath>
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
        const float3_64 refEfield = float3_64(0.1727409809769384, -0.008861178965595797, 0.09933584040072169);
        float3_64 const refBfield = float3_64(3.3267856482525846e-10, 5.027189941003478e-11, -5.739801666328101e-10);
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

// This combination of compilers has a bug that is triggered by Catch2 internally suppressing warnings.
// See https://github.com/ComputationalRadiationPhysics/picongpu/pull/5174#issuecomment-2467890326
#if (__GNUC__ != 11 || __CUDACC_VER_MAJOR__ != 11)
/** Check the cutoff against the envelope geometry in SI coordinates.
 *
 * The center satisfies y = beta*c*t - z*(beta-cos(phi))/sin(phi).
 * The Gaussian width follows independently from q = tan(alpha)^2*z/(c*sin(phi)),
 * avoiding the guard's nu/xi expression when constructing the expected boundary.
 */
struct twtsTightEnvelopeGuardTest
{
    static constexpr float_64 wavelength = 800.0e-9;
    static constexpr float_64 pulseLength = 30.0e-15;
    // Keep individual Bessel values representable in float in the end-to-end check by using a small waist.
    static constexpr float_64 waist = 0.5e-6;

    static void checkEnvelope(float_64 const beta, float_64 const phi)
    {
        templates::twtstight::EField const
            eField{0.0, wavelength, pulseLength, waist, phi, beta, 0.0, false, 0.0, PI / 6.0};
        templates::twtstight::BField const
            bField{0.0, wavelength, pulseLength, waist, phi, beta, 0.0, false, 0.0, PI / 6.0};
        float_64 const c = sim.si.getSpeedOfLight();
        float_64 const sinPhi = std::sin(std::abs(phi));
        float_64 const cosPhi = std::cos(phi);
        float_64 const tanAlpha = (1.0 - beta * cosPhi) / (beta * sinPhi);
        float_64 const tau = 2.0 * pulseLength;
        float_64 const omega = 2.0 * PI * c / wavelength;
        float_64 const period = wavelength / (c * (1.0 - beta * cosPhi));
        float_64 const phiSign = phi < 0.0 ? -1.0 : 1.0;

        // Include no broadening and both signs of the imaginary part of sigma.
        for(float_64 const broadening : {-2.0, 0.0, 2.0})
        {
            float_64 const z = broadening * omega * tau * tau * c * sinPhi / (2.0 * tanAlpha * tanAlpha);
            float_64 const width = tau * std::sqrt(1.0 + broadening * broadening);
            // The second time exercises the reduced-coordinate translation by many periods.
            for(float_64 const periods : {0.25, 17.25})
            {
                float_64 const time = periods * period;
                float_64 const centerY = beta * c * time - z * (beta - cosPhi) / sinPhi;
                // Stay clear of equality: coordinates are rounded to float_T by the functor.
                for(float_64 const offset : {-1.05, -0.95, 0.0, 0.95, 1.05})
                {
                    float_64 const u = offset * templates::twtstight::numSigmas * width;
                    float3_64 const pos{0.0, centerY - beta * c * u, phiSign * z};
                    bool const outside = std::abs(offset) > 1.0;
                    INFO("beta = " << beta << ", phi = " << phi << ", broadening = " << broadening);
                    INFO("periods = " << periods << ", boundary fraction = " << offset);
                    CHECK(eField.isOutsideTWTSEnvelope(eField.defineMinimalCoordinates(pos, time)) == outside);
                    CHECK(bField.isOutsideTWTSEnvelope(bField.defineMinimalCoordinates(pos, time)) == outside);
                }
            }
        }
    }

    /** Exercise the actual field early-outs, including the negative-angle coordinate rotation. */
    template<typename T_Field>
    static void checkNegativePhi(float_64 const beta)
    {
        using float_T = templates::twtstight::float_T;
        float_64 const phi = -PI / 6.0;
        T_Field const field{0.0, wavelength, pulseLength, waist, phi, beta, 0.0, false, 0.0, PI / 6.0};
        float_64 const c = sim.si.getSpeedOfLight();
        float_64 const sinPhi = std::sin(-phi);
        float_64 const cosPhi = std::cos(phi);
        float_64 const tanAlpha = (1.0 - beta * cosPhi) / (beta * sinPhi);
        float_64 const tau = 2.0 * pulseLength;
        float_64 const omega = 2.0 * PI * c / wavelength;
        float_64 const z = 300.0e-6;
        float_64 const broadening = 2.0 * tanAlpha * tanAlpha * z / (c * sinPhi * omega * tau * tau);
        float_64 const width = tau * std::sqrt(1.0 + broadening * broadening);
        float_64 const time = 17.25 * wavelength / (c * (1.0 - beta * cosPhi));
        float_64 const centerY = beta * c * time - z * (beta - cosPhi) / sinPhi;

        INFO("negative-angle field check, beta = " << beta);
        for(float_64 const offset : {-1.05, 0.0, 1.05})
        {
            float_64 const u = offset * templates::twtstight::numSigmas * width;
            float3_64 const pos{-0.1e-6, centerY - beta * c * u, -z};
            std::array<float_T, 3u> const values{
                field.calcTWTSFieldX(pos, time),
                field.calcTWTSFieldY(pos, time),
                field.calcTWTSFieldZ(pos, time)};
            INFO("boundary fraction = " << offset);
            bool anyNonzero = false;
            for(auto const value : values)
            {
                CHECK(std::isfinite(value));
                anyNonzero = anyNonzero || value != float_T(0.0);
                if(offset != 0.0)
                    CHECK(value == float_T(0.0));
            }
            // Test the vector rather than requiring each oscillatory component to be nonzero.
            if(offset == 0.0)
                CHECK(anyNonzero);
        }
    }
};
#endif

TEST_CASE("unit::TWTSTightEnvelopeGuard", "[TWTSTight laser envelope guard test]")
{
#if (__GNUC__ != 11 || __CUDACC_VER_MAJOR__ != 11)
    for(float_64 const beta : {0.9, 1.0, 1.1})
    {
        for(float_64 const phi : {-PI / 2.0, -PI / 6.0, PI / 6.0, PI / 2.0})
            twtsTightEnvelopeGuardTest::checkEnvelope(beta, phi);
        twtsTightEnvelopeGuardTest::checkNegativePhi<templates::twtstight::EField>(beta);
        twtsTightEnvelopeGuardTest::checkNegativePhi<templates::twtstight::BField>(beta);
    }
#endif
}
