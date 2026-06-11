/*
 * SPDX-FileCopyrightText: Tapish Narwal
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <pmacc/boost_workaround.hpp>

#include "picongpu/plugins/binning/FunctorDescription.hpp"
#include "picongpu/plugins/binning/axis/LinearAxis.hpp"
#include "picongpu/plugins/binning/axis/LogAxis.hpp"

#include <string>
#include <type_traits>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

using namespace picongpu::plugins::binning;

TEST_CASE("axis::Range Construction Validation", "[axis][Range]")
{
    SECTION("Double Type")
    {
        CHECK_NOTHROW(axis::Range<double>{0.0, 10.0});
        CHECK_NOTHROW(axis::Range<double>{-10.0, -5.0});
        CHECK_THROWS(axis::Range<double>{5.0, 5.0});
        CHECK_THROWS(axis::Range<double>{10.0, 0.0});
        CHECK_THROWS(axis::Range<double>{-5.0, -10.0});
    }

    SECTION("Integer Type")
    {
        CHECK_NOTHROW(axis::Range<int>{0, 10});
        CHECK_NOTHROW(axis::Range<int>{-10, -5});
        CHECK_THROWS(axis::Range<int>{5, 5});
        CHECK_THROWS(axis::Range<int>{10, 0});
        CHECK_THROWS(axis::Range<int>{-5, -10});
    }

    SECTION("Unsigned Integer Type")
    {
        CHECK_NOTHROW(axis::Range<unsigned int>{0, 10});
        CHECK_THROWS(axis::Range<unsigned int>{5, 5});
        CHECK_THROWS(axis::Range<unsigned int>{10, 0});
    }
}

template<typename TValue>
struct AxisEdgeTestCase
{
    std::string description;
    axis::AxisSplitting<TValue> split;
    std::vector<double> expectedEdges;
    bool expectThrow = false;

    auto getMatcher() const
    {
        if constexpr(std::is_floating_point_v<TValue>)
        {
            return Catch::Matchers::Approx(expectedEdges).epsilon(1e-9);
        }
        else
        {
            // return Catch::Matchers::Equals(expectedEdges);
            return Catch::Matchers::Approx(expectedEdges).epsilon(1e-9);
        }
    }
};

auto identityFunctor = [](auto val) { return val; };

template<typename TValue, typename AxisCreatorFunc>
void runSingleAxisEdgeTest(
    AxisEdgeTestCase<TValue> const& testCase,
    AxisCreatorFunc createAxis,
    auto const& functorDesc)
{
    if(testCase.expectThrow)
    {
        REQUIRE_THROWS(createAxis(testCase.split, functorDesc));
    }
    else
    {
        auto axis = createAxis(testCase.split, functorDesc);
        auto binEdges = axis.getBinEdgesSI();
        CHECK_THAT(binEdges, testCase.getMatcher());

        auto expectedNumBins = testCase.split.nBins;
        if(testCase.split.enableOverflowBins)
        {
            expectedNumBins += 2;
        }
        CHECK(axis.getNBins() == expectedNumBins);

        CHECK(axis.label == functorDesc.name);

        using Catch::Matchers::RangeEquals;
        CHECK_THAT(axis.units, RangeEquals(functorDesc.units));
        CHECK(axis.getUnitConversion() == Catch::Approx(getConversionFactor(functorDesc.units)).epsilon(1e-9));
    }
}

template<
    typename TValue,
    typename AxisCreatorFunc,
    typename TFunctorDesc = FunctorDescription<TValue, decltype(identityFunctor)>>
void runAxisEdgesTestCases(
    std::string const& valueTypeDesc,
    std::vector<AxisEdgeTestCase<TValue>> const& testCases,
    AxisCreatorFunc createAxis,
    TFunctorDesc const& functorDesc = createFunctorDescription<TValue>(identityFunctor, "default_dimensionless"))
{
    for(auto const& testCase : testCases)
    {
        SECTION(valueTypeDesc + " [" + functorDesc.name + "]: " + testCase.description)
        {
            runSingleAxisEdgeTest(testCase, createAxis, functorDesc);
        }
    }
}

// --- Test Cases for Axis::getBinEdgesSI ---

TEST_CASE("LinearAxis Bin Edges", "[axis][LinearAxis][Edges]")
{
    auto createLinear = [](auto const& split, auto const& funcDesc) { return axis::createLinear(split, funcDesc); };
    SECTION("Double Type")
    {
        std::vector<AxisEdgeTestCase<double>> const linearDoubleTestCases
            = {{"Range half bins",
                {{0.0, 10.0}, 20, false},
                {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0},
                false},
               {"Negative Range",
                {{-10.0, -1.0}, 9, false},
                {-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0},
                false},
               {"Range Crossing Zero",
                {{-5.0, 5.0}, 10, false},
                {-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0},
                false}};

        runAxisEdgesTestCases("Double", linearDoubleTestCases, createLinear);
    }
    SECTION("Integer Type")
    {
        std::vector<AxisEdgeTestCase<int>> const linearIntTestCases
            = {{"Range (nBins = range)", {{0, 10}, 10, false}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, false},
               {"Range (nBins > range)",
                {{0, 10}, 20, false},
                {0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10},
                false},
               {"Range (Exact Divisibility)", {{0, 12}, 6, false}, {0, 2, 4, 6, 8, 10, 12}, false},
               {"Range (Non-Divisible)", {{0, 7}, 5, false}, {0.0, 1.4, 2.8, 4.2, 5.6, 7.0}, false},
               {"Negative Range", {{-10, -1}, 9, false}, {-10, -9, -8, -7, -6, -5, -4, -3, -2, -1}, false},
               {"Range Crossing Zero", {{-5, 5}, 10, false}, {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}, false}};

        runAxisEdgesTestCases("Integer", linearIntTestCases, createLinear);
    }
}

TEST_CASE("LogAxis Bin Edges", "[axis][LogAxis][Edges]")
{
    auto createLog = [](auto const& split, auto const& funcDesc) { return axis::createLog(split, funcDesc); };
    SECTION("Double Type")
    {
        std::vector<AxisEdgeTestCase<double>> const logDoubleTestCases
            = {{"Range 5", {{0.5, 16.0}, 5, false}, {0.5, 1.0, 2.0, 4.0, 8.0, 16.0}, false},
               {"Range Negative", {{-9.0, -1.0}, 2, false}, {-9.0, -3.0, -1.0}, false},
               {"Range crosses 0 - Expect Throw", {{-1.0, 1.0}, 4, false}, {}, true},
               {"Range (min is 0) - Expect Throw", {{0.0, 1.0}, 2, false}, {}, true}};

        runAxisEdgesTestCases("Double", logDoubleTestCases, createLog);
    }
    SECTION("Integer Type")
    {
        std::vector<AxisEdgeTestCase<int>> const logIntTestCases
            = {{"Range (Exact Power of 2 Ratio)", {{1, 32}, 5, false}, {1, 2, 4, 8, 16, 32}, false},
               {"Range 3", {{1, 27}, 3, false}, {1, 3, 9, 27}, false},
               {"Range Negative", {{-16, -1}, 4, false}, {-16, -8, -4, -2, -1}, false},
               {"Range (Non-Integral bins)",
                {{1, 27}, 5, false},
                {1, std::pow(27, 1. / 5), std::pow(27, 2. / 5), std::pow(27, 3. / 5), std::pow(27, 4. / 5), 27.0},
                false}};

        runAxisEdgesTestCases("Integer", logIntTestCases, createLog);
    }
}

TEST_CASE("LinearAxis Bin Edges with Units", "[axis][LinearAxis][Edges][Units]")
{
    constexpr std::array<double, picongpu::plugins::binning::numUnits> velocityUnits
        = {1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0};

    constexpr double c = 299792458.0;

    std::vector<AxisEdgeTestCase<double>> const linearVelocityTestCases
        = {{"Velocity [0, c], 3 bins", {{0.0, c}, 3, false}, {0.0, c / 3.0, 2.0 * c / 3.0, c}, false},
           {"Velocity [-c/2, c/2], 4 bins",
            {{-c / 2.0, c / 2.0}, 4, false},
            {-c / 2.0, -c / 4.0, 0.0, c / 4.0, c / 2.0},
            false},
           {"Velocity [0.1c, 0.9c], 8 bins, overflow",
            {{0.1 * c, 0.9 * c}, 8, true},
            {0.1 * c, 0.2 * c, 0.3 * c, 0.4 * c, 0.5 * c, 0.6 * c, 0.7 * c, 0.8 * c, 0.9 * c},
            false}};
    auto const velocityFunctorDesc = createFunctorDescription<double>(identityFunctor, "Velocity", velocityUnits);
    auto createLinear = [](auto const& split, auto const& funcDesc) { return axis::createLinear(split, funcDesc); };
    runAxisEdgesTestCases("Double", linearVelocityTestCases, createLinear, velocityFunctorDesc);
}

TEST_CASE("LogAxis Bin Edges with Units", "[axis][LogAxis][Edges][Units]")
{
    constexpr std::array<double, picongpu::plugins::binning::numUnits> velocityUnits
        = {1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0};
    constexpr double c = 299792458.0;

    std::vector<AxisEdgeTestCase<double>> const logVelocityTestCases = {
        {"Velocity [0.01c, 0.99c], 4 bins, overflow",
         {{0.0001 * c, 1 * c}, 4, true},
         {0.0001 * c, 0.001 * c, 0.01 * c, 0.1 * c, 1 * c},
         false},
        {"Velocity [-0.9c, -0.1c], 2 bins, overflow",
         {{-0.9 * c, -0.1 * c}, 2, true},
         {-0.9 * c, -0.3 * c, -0.1 * c},
         false},
    };
    auto const velocityFunctorDesc = createFunctorDescription<double>(identityFunctor, "Velocity", velocityUnits);
    auto createLog = [](auto const& split, auto const& funcDesc) { return axis::createLog(split, funcDesc); };
    runAxisEdgesTestCases("Double", logVelocityTestCases, createLog, velocityFunctorDesc);
}

// --- Test Cases for Axis::getBinIdx ---

template<typename TValue>
struct BinIdxTestCase
{
    std::string description;
    TValue inputValue;
    // { shouldBin, binIndex }
    std::pair<bool, uint32_t> expectedResult;
};

template<typename TValue, typename TKernel>
void runBinIdxTests(TKernel const& kernel, std::vector<BinIdxTestCase<TValue>> const& testCases)
{
    for(auto const& tc : testCases)
    {
        SECTION(tc.description)
        {
            auto result = kernel.getBinIdx(tc.inputValue);
            auto [shouldBin, binIndex] = result;

            REQUIRE(shouldBin == tc.expectedResult.first);
            if(shouldBin)
            {
                REQUIRE(binIndex == tc.expectedResult.second);
            }
        }
    }
}

TEST_CASE("LogAxisKernel::getBinIdx", "[axis][LogAxis][Kernel]")
{
    SECTION("Double Type")
    {
        SECTION("Overflow Enabled, Positive Range [1.0, 16.0], 4 bins")
        {
            // Total bins = 4 + 2 = 6. Bins: 0 (under), 1-4 (in range), 5 (over)
            // Edges approx: 1.0, 2.0, 4.0, 8.0, 16.0
            axis::AxisSplitting<double> split{{1.0, 16.0}, 4, true};
            auto axis = axis::createLog(split, createFunctorDescription<double>(identityFunctor, "test"));
            auto kernel = axis.getAxisKernel();

            // Note: scaling = 4 / log2(16.0/1.0) = 4 / log2(16) = 4 / 4 = 1.0
            // Formula (approx): floor( log2(val / 1.0) * 1.0 ) + 1  (for overflow=true)
            // binIdx = floor(log2(val)) + 1

            std::vector<BinIdxTestCase<double>> tests = {
                {"Value below min", 0.5, {true, 0}},
                {"Value exactly min", 1.0, {true, 1}}, // log2(1)=0 -> floor(0)+1 = 1
                {"Value in first bin", 1.9, {true, 1}}, // log2(1.9)~0.9 -> floor(0.9)+1 = 1
                {"Value exactly second edge", 2.0, {true, 2}}, // log2(2)=1 -> floor(1)+1 = 2
                {"Value in second bin", 3.9, {true, 2}}, // log2(3.9)~1.96 -> floor(1.96)+1 = 2
                {"Value exactly third edge", 4.0, {true, 3}}, // log2(4)=2 -> floor(2)+1 = 3
                {"Value in third bin", 7.9, {true, 3}}, // log2(7.9)~2.98 -> floor(2.98)+1 = 3
                {"Value exactly fourth edge", 8.0, {true, 4}}, // log2(8)=3 -> floor(3)+1 = 4
                {"Value in fourth bin", 15.9, {true, 4}}, // log2(15.9)~3.99 -> floor(3.99)+1 = 4
                {"Value exactly max", 16.0, {true, 5}}, // val >= max goes to overflow
                {"Value above max", 20.0, {true, 5}},
                {"Large value", 1e6, {true, 5}},
                {"Small positive value", 1e-6, {true, 0}},
            };
            runBinIdxTests(kernel, tests);
        }

        SECTION("Overflow Disabled, Positive Range [1.0, 16.0], 4 bins")
        {
            // Total bins = 4. Bins: 0-3 (in range)
            // Edges approx: 1.0, 2.0, 4.0, 8.0, 16.0
            axis::AxisSplitting<double> split{{1.0, 16.0}, 4, false};
            auto axis = axis::createLog(split, createFunctorDescription<double>(identityFunctor, "test"));
            auto kernel = axis.getAxisKernel();

            // Note: scaling = 1.0
            // Formula (approx): floor( log2(val / 1.0) * 1.0 )  (for overflow=false)
            // binIdx = floor(log2(val))

            std::vector<BinIdxTestCase<double>> tests = {
                {"Value below min", 0.5, {false, {}}}, // Should not bin, index 0
                {"Value exactly min", 1.0, {true, 0}}, // log2(1)=0 -> floor(0)=0
                {"Value in first bin", 1.9, {true, 0}}, // log2(1.9)~0.9 -> floor(0.9)=0
                {"Value exactly second edge", 2.0, {true, 1}}, // log2(2)=1 -> floor(1)=1
                {"Value in second bin", 3.9, {true, 1}}, // log2(3.9)~1.96 -> floor(1.96)=1
                {"Value exactly third edge", 4.0, {true, 2}}, // log2(4)=2 -> floor(2)=2
                {"Value in third bin", 7.9, {true, 2}}, // log2(7.9)~2.98 -> floor(2.98)=2
                {"Value exactly fourth edge", 8.0, {true, 3}}, // log2(8)=3 -> floor(3)=3
                {"Value in fourth bin", 15.9, {true, 3}}, // log2(15.9)~3.99 -> floor(3.99)=3
                {"Value exactly max", 16.0, {false, {}}}, // Should not bin, index nBins-1 = 3
                {"Value above max", 20.0, {false, {}}}, // Should not bin, index nBins-1 = 3
            };
            runBinIdxTests(kernel, tests);
        }

        SECTION("Overflow Enabled, Negative Range [-16.0, -1.0], 4 bins")
        {
            // Total bins = 4 + 2 = 6. Bins: 0 (underflow |val| > |-16|), 1-4 (in range), 5 (overflow |val| < |-1|)
            // Edges approx: -16.0, -8.0, -4.0, -2.0, -1.0
            // picRange.min = -16.0, picRange.max = -1.0
            axis::AxisSplitting<double> split{{-16.0, -1.0}, 4, true};
            auto axis = axis::createLog(split, createFunctorDescription<double>(identityFunctor, "test"));
            auto kernel = axis.getAxisKernel();

            // Note: scaling = 4 / log2(-1.0 / -16.0) = 4 / log2(1/16) = 4 / (-4) = -1.0
            // Formula (approx): floor( log2(val / (-16.0)) * (-1.0) ) + 1
            // binIdx = floor(-log2(val / -16.0)) + 1 = floor(log2(-16.0 / val)) + 1

            std::vector<BinIdxTestCase<double>> tests = {
                {"Value 'below' min (more negative)", -20.0, {true, 0}},
                {"Value exactly min", -16.0, {true, 1}}, // log2(-16/-16)=log2(1)=0 -> floor(0)+1 = 1
                {"Value in first bin", -15.9, {true, 1}}, // log2(-16/-15.9)~log2(1.006)~0.009 -> floor(0.009)+1 = 1
                {"Value exactly second edge", -8.0, {true, 2}}, // log2(-16/-8)=log2(2)=1 -> floor(1)+1 = 2
                {"Value in second bin", -7.9, {true, 2}}, // log2(-16/-7.9)~log2(2.02)~1.01 -> floor(1.01)+1 = 2
                {"Value exactly third edge", -4.0, {true, 3}}, // log2(-16/-4)=log2(4)=2 -> floor(2)+1 = 3
                {"Value in third bin", -3.9, {true, 3}}, // log2(-16/-3.9)~log2(4.1)~2.03 -> floor(2.03)+1 = 3
                {"Value exactly fourth edge", -2.0, {true, 4}}, // log2(-16/-2)=log2(8)=3 -> floor(3)+1 = 4
                {"Value in fourth bin", -1.1, {true, 4}}, // log2(-16/-1.1)~log2(14.5)~3.86 -> floor(3.86)+1 = 4
                {"Value exactly max", -1.0, {true, 5}}, // val >= max goes to overflow bin
                {"Value 'above' max (less negative)", -0.5, {true, 5}},
                {"Small negative value", -1e-6, {true, 5}},
            };
            runBinIdxTests(kernel, tests);
        }
    }

    SECTION("Integer Type")
    {
        // Use a range where calculations are exact powers
        SECTION("Overflow Enabled, Positive Range [1, 27], 3 bins")
        {
            // Total bins = 3 + 2 = 5. Bins: 0 (under), 1-3 (in range), 4 (over)
            // Edges: 1, 3, 9, 27
            axis::AxisSplitting<int> split{{1, 27}, 3, true};
            auto axis = axis::createLog(split, createFunctorDescription<int>(identityFunctor, "test"));
            auto kernel = axis.getAxisKernel();

            // Note: scaling = 3 / log2(27.0/1.0) = 3 / log2(27) approx 3 / 4.75 = 0.63 (float)
            // Using double for calculation:
            // ScalingType = float_X -> float
            // scaling = static_cast<float>(3) / static_cast<float>(std::log2(27.0 / 1.0))
            // scaling approx 3.0f / 4.7548876f = 0.63092977f
            // Formula (approx): floor( log2(val / 1.0) * scaling ) + 1
            // binIdx = floor(log2(val) * 0.63092977f) + 1

            std::vector<BinIdxTestCase<int>> tests = {
                {"Value below min", 0, {true, 0}},
                {"Value exactly min", 1, {true, 1}}, // log2(1)=0 -> floor(0*s)+1=1
                {"Value in first bin", 2, {true, 1}}, // log2(2)=1 -> floor(1*s)+1=floor(0.63)+1=1
                {"Value exactly second edge", 3, {true, 2}}, // log2(3)~1.58 -> floor(1.58*s)+1=floor(0.9999)+1=1 -
                // This case can have floating point issues
            };
            runBinIdxTests(kernel, tests); // Skip this specific case due to potential precision issues with
            // non-power-of-2 ranges + integer types
        }

        SECTION("Overflow Enabled, Positive Range [1, 32], 5 bins")
        {
            // Total bins = 5 + 2 = 7. Bins: 0 (under), 1-5 (in range), 6 (over)
            // Edges: 1, 2, 4, 8, 16, 32
            axis::AxisSplitting<int> split{{1, 32}, 5, true};
            auto axis = axis::createLog(split, createFunctorDescription<int>(identityFunctor, "test"));
            auto kernel = axis.getAxisKernel();

            // ScalingType = float_X -> float
            // scaling = static_cast<float>(5) / static_cast<float>(std::log2(32.0 / 1.0))
            // scaling = 5.0f / 5.0f = 1.0f
            // Formula: floor( log2(val / 1.0) * 1.0f ) + 1
            // binIdx = floor(log2(val)) + 1

            std::vector<BinIdxTestCase<int>> tests = {
                {"Value below min", 0, {true, 0}},
                {"Value exactly min", 1, {true, 1}}, // log2(1)=0 -> floor(0)+1=1
                {"Value in first bin", 1, {true, 1}}, // Re-test min
                {"Value exactly second edge", 2, {true, 2}}, // log2(2)=1 -> floor(1)+1=2
                {"Value in second bin", 3, {true, 2}}, // log2(3)~1.58 -> floor(1.58)+1=2
                {"Value exactly third edge", 4, {true, 3}}, // log2(4)=2 -> floor(2)+1=3
                {"Value in third bin", 7, {true, 3}}, // log2(7)~2.8 -> floor(2.8)+1=3
                {"Value exactly fourth edge", 8, {true, 4}}, // log2(8)=3 -> floor(3)+1=4
                {"Value in fourth bin", 15, {true, 4}}, // log2(15)~3.9 -> floor(3.9)+1=4
                {"Value exactly fifth edge", 16, {true, 5}}, // log2(16)=4 -> floor(4)+1=5
                {"Value in fifth bin", 31, {true, 5}}, // log2(31)~4.95 -> floor(4.95)+1=5
                {"Value exactly max", 32, {true, 6}}, // >= max goes to overflow
                {"Value above max", 100, {true, 6}},
            };
            runBinIdxTests(kernel, tests);
        }

        SECTION("Overflow Disabled, Positive Range [1, 32], 5 bins")
        {
            // Total bins = 5. Bins: 0-4 (in range)
            axis::AxisSplitting<int> split{{1, 32}, 5, false};
            auto axis = axis::createLog(split, createFunctorDescription<int>(identityFunctor, "test"));
            auto kernel = axis.getAxisKernel();

            // scaling = 1.0f
            // Formula: floor( log2(val / 1.0) * 1.0f )
            // binIdx = floor(log2(val))

            std::vector<BinIdxTestCase<int>> tests = {
                {"Value below min", 0, {false, {}}}, // Should not bin, index 0
                {"Value exactly min", 1, {true, 0}}, // log2(1)=0 -> floor(0)=0
                {"Value in first bin", 1, {true, 0}},
                {"Value exactly second edge", 2, {true, 1}}, // log2(2)=1 -> floor(1)=1
                {"Value in second bin", 3, {true, 1}},
                {"Value exactly third edge", 4, {true, 2}}, // log2(4)=2 -> floor(2)=2
                {"Value in third bin", 7, {true, 2}},
                {"Value exactly fourth edge", 8, {true, 3}}, // log2(8)=3 -> floor(3)=3
                {"Value in fourth bin", 15, {true, 3}},
                {"Value exactly fifth edge", 16, {true, 4}}, // log2(16)=4 -> floor(4)=4
                {"Value in fifth bin", 31, {true, 4}},
                {"Value exactly max", 32, {false, {}}}, // Should not bin, index nBins-1 = 4
                {"Value above max", 100, {false, {}}}, // Should not bin, index nBins-1 = 4
            };
            runBinIdxTests(kernel, tests);
        }

        SECTION("Overflow Enabled, Positive Range [1, 61], 10 bins")
        {
            // Total bins = 10 + 2 = 12. Bins: 0 (under), 1-10 (in range), 11 (over)
            // Edges approx: 1, 1.50, 2.25, 3.37, 5.06, 7.59, 11.39, 17.09, 25.64, 38.47, 61
            axis::AxisSplitting<int> split{{1, 61}, 10, true}; // max=61 to make range size 60
            auto axis = axis::createLog(split, createFunctorDescription<int>(identityFunctor, "test"));
            auto kernel = axis.getAxisKernel();

            // ScalingType = float_X -> float
            // scaling = static_cast<float>(10) / static_cast<float>(std::log2(61.0 / 1.0))
            // scaling = 10.0f / log2(61.0f) approx 10.0f / 5.9307f = 1.6861f
            // Formula: floor( log2(val / 1.0) * scaling ) + 1
            // binIdx = floor(log2(val) * 1.6861f) + 1

            std::vector<BinIdxTestCase<int>> tests = {
                {"Value below min", 0, {true, 0}},
                {"Value exactly min", 1, {true, 1}}, // floor(log2(1)*s)+1 = floor(0)+1 = 1
                {"Value in first bin", 1, {true, 1}}, // Re-test min
                {"Value in second bin", 2, {true, 2}}, // floor(log2(2)*s)+1 = floor(1*1.6861)+1 = 1+1 = 2
                {"Value in third bin", 3, {true, 3}}, // floor(log2(3)*s)+1 = floor(1.58*s)+1 = floor(2.67)+1 = 2+1 = 3
                {"Value in fourth bin",
                 5,
                 {true, 4}}, // floor(log2(5)*s)+1 = floor(2.32*s)+1 = floor(3.91)+1 = 3+1 = 4
                {"Value in fifth bin", 7, {true, 5}}, // floor(log2(7)*s)+1 = floor(2.81*s)+1 = floor(4.73)+1 = 4+1 = 5
                {"Value in sixth bin",
                 11,
                 {true, 6}}, // floor(log2(11)*s)+1 = floor(3.46*s)+1 = floor(5.83)+1 = 5+1 = 6
                {"Value in seventh bin",
                 17,
                 {true, 7}}, // floor(log2(17)*s)+1 = floor(4.09*s)+1 = floor(6.89)+1 = 6+1 = 7
                {"Value in eighth bin",
                 25,
                 {true, 8}}, // floor(log2(25)*s)+1 = floor(4.64*s)+1 = floor(7.83)+1 = 7+1 = 8
                {"Value in ninth bin",
                 38,
                 {true, 9}}, // floor(log2(38)*s)+1 = floor(5.25*s)+1 = floor(8.84)+1 = 8+1 = 9
                {"Value in tenth bin",
                 60,
                 {true, 10}}, // floor(log2(60)*s)+1 = floor(5.91*s)+1 = floor(9.96)+1 = 9+1 = 10
                {"Value exactly max", 61, {true, 11}}, // >= max goes to overflow
                {"Value above max", 100, {true, 11}},
            };
            runBinIdxTests(kernel, tests);
        }
    }
}

TEST_CASE("LinearAxisKernel::getBinIdx", "[axis][LinearAxis][Kernel]")
{
    SECTION("Double Type")
    {
        SECTION("Overflow Enabled, Positive Range [0.0, 10.0], 10 bins")
        {
            // Total bins = 10 + 2 = 12. Bins: 0 (under), 1-10 (in range), 11 (over)
            // Edges: 0.0, 1.0, 2.0, ..., 10.0
            // Bin width = 1.0. scaling = 10 / (10.0 - 0.0) = 1.0
            // Formula: floor((val - 0.0) * 1.0) + 1 = floor(val) + 1
            axis::AxisSplitting<double> split{{0.0, 10.0}, 10, true};
            auto axis = axis::createLinear(split, createFunctorDescription<double>(identityFunctor, "test"));
            auto kernel = axis.getAxisKernel();

            std::vector<BinIdxTestCase<double>> tests = {
                {"Value below min", -1.0, {true, 0}},
                {"Value exactly min", 0.0, {true, 1}}, // floor(0)+1 = 1
                {"Value in first bin", 0.5, {true, 1}}, // floor(0.5)+1 = 1
                {"Value near first edge upper bound", 0.999999, {true, 1}}, // floor(~1)+1 = 1
                {"Value exactly second edge", 1.0, {true, 2}}, // floor(1.0)+1 = 2
                {"Value in second bin", 1.5, {true, 2}}, // floor(1.5)+1 = 2
                {"Value in last bin", 9.5, {true, 10}}, // floor(9.5)+1 = 10
                {"Value near max", 9.999999, {true, 10}}, // floor(~10)+1 = 10
                {"Value exactly max", 10.0, {true, 11}}, // >= max goes to overflow
                {"Value above max", 11.0, {true, 11}},
                {"Large value", 1e6, {true, 11}},
                {"Small negative value", -1e-6, {true, 0}},
            };
            runBinIdxTests(kernel, tests);
        }

        SECTION("Overflow Disabled, Positive Range [0.0, 10.0], 10 bins")
        {
            // Total bins = 10. Bins: 0-9 (in range)
            // Edges: 0.0, 1.0, 2.0, ..., 10.0
            // scaling = 1.0
            // Formula: floor((val - 0.0) * 1.0) = floor(val)
            axis::AxisSplitting<double> split{{0.0, 10.0}, 10, false};
            auto axis = axis::createLinear(split, createFunctorDescription<double>(identityFunctor, "test"));
            auto kernel = axis.getAxisKernel();

            std::vector<BinIdxTestCase<double>> tests = {
                {"Value below min", -1.0, {false, {}}}, // Should not bin, index 0
                {"Value exactly min", 0.0, {true, 0}}, // floor(0) = 0
                {"Value in first bin", 0.5, {true, 0}}, // floor(0.5) = 0
                {"Value near first edge upper bound", 0.999999, {true, 0}}, // floor(~1) = 0
                {"Value exactly second edge", 1.0, {true, 1}}, // floor(1.0) = 1
                {"Value in second bin", 1.5, {true, 1}}, // floor(1.5) = 1
                {"Value in last bin", 9.5, {true, 9}}, // floor(9.5) = 9
                {"Value near max", 9.999999, {true, 9}}, // floor(~10) = 9
                {"Value exactly max", 10.0, {false, {}}}, // Should not bin, index nBins-1 = 9
                {"Value above max", 11.0, {false, {}}}, // Should not bin, index nBins-1 = 9
            };
            runBinIdxTests(kernel, tests);
        }

        SECTION("Overflow Enabled, Negative Range [-10.0, -5.0], 5 bins")
        {
            // Total bins = 5 + 2 = 7. Bins: 0 (under), 1-5 (in range), 6 (over)
            // Edges: -10.0, -9.0, -8.0, -7.0, -6.0, -5.0
            // Bin width = 1.0. scaling = 5 / (-5.0 - (-10.0)) = 5 / 5.0 = 1.0
            // Formula: floor((val - (-10.0)) * 1.0) + 1 = floor(val + 10.0) + 1
            axis::AxisSplitting<double> split{{-10.0, -5.0}, 5, true};
            auto axis = axis::createLinear(split, createFunctorDescription<double>(identityFunctor, "test"));
            auto kernel = axis.getAxisKernel();

            std::vector<BinIdxTestCase<double>> tests = {
                {"Value below min", -11.0, {true, 0}},
                {"Value exactly min", -10.0, {true, 1}}, // floor(-10+10)+1 = 1
                {"Value in first bin", -9.5, {true, 1}}, // floor(-9.5+10)+1 = floor(0.5)+1 = 1
                {"Value exactly second edge", -9.0, {true, 2}}, // floor(-9+10)+1 = floor(1)+1 = 2
                {"Value in last bin", -5.1, {true, 5}}, // floor(-5.1+10)+1 = floor(4.9)+1 = 5
                {"Value exactly max", -5.0, {true, 6}}, // >= max goes to overflow
                {"Value above max", -4.0, {true, 6}},
                {"Value above max (positive)", 1.0, {true, 6}},
            };
            runBinIdxTests(kernel, tests);
        }
    }

    SECTION("Integer Type")
    {
        SECTION("Overflow Enabled, Positive Range [0, 12], 6 bins")
        {
            // Total bins = 6 + 2 = 8. Bins: 0 (under), 1-6 (in range), 7 (over)
            // Edges: 0, 2, 4, 6, 8, 10, 12
            // Bin width = (12-0)/6 = 2.
            // scaling = static_cast<float>(6) / (12 - 0) = 6.0f / 12.0f = 0.5f
            // Formula: floor((val - 0) * 0.5f) + 1 = floor(val * 0.5f) + 1
            axis::AxisSplitting<int> split{{0, 12}, 6, true};
            auto axis = axis::createLinear(split, createFunctorDescription<int>(identityFunctor, "test"));
            auto kernel = axis.getAxisKernel();

            std::vector<BinIdxTestCase<int>> tests = {
                {"Value below min", -1, {true, 0}},
                {"Value exactly min", 0, {true, 1}}, // floor(0*0.5)+1 = 1
                {"Value in first bin", 1, {true, 1}}, // floor(1*0.5)+1 = floor(0.5)+1 = 1
                {"Value exactly second edge", 2, {true, 2}}, // floor(2*0.5)+1 = floor(1.0)+1 = 2
                {"Value in second bin", 3, {true, 2}}, // floor(3*0.5)+1 = floor(1.5)+1 = 2
                {"Value exactly third edge", 4, {true, 3}}, // floor(4*0.5)+1 = floor(2.0)+1 = 3
                {"Value in last bin (low)", 10, {true, 6}}, // floor(10*0.5)+1 = floor(5.0)+1 = 6
                {"Value in last bin (high)", 11, {true, 6}}, // floor(11*0.5)+1 = floor(5.5)+1 = 6
                {"Value exactly max", 12, {true, 7}}, // >= max goes to overflow
                {"Value above max", 13, {true, 7}},
            };
            runBinIdxTests(kernel, tests);
        }

        SECTION("Overflow Disabled, Positive Range [0, 12], 6 bins")
        {
            // Total bins = 6. Bins: 0-5 (in range)
            // Edges: 0, 2, 4, 6, 8, 10, 12
            // scaling = 0.5f
            // Formula: floor((val - 0) * 0.5f) = floor(val * 0.5f)
            axis::AxisSplitting<int> split{{0, 12}, 6, false};
            auto axis = axis::createLinear(split, createFunctorDescription<int>(identityFunctor, "test"));
            auto kernel = axis.getAxisKernel();

            std::vector<BinIdxTestCase<int>> tests = {
                {"Value below min", -1, {false, {}}},
                {"Value exactly min", 0, {true, 0}}, // floor(0*0.5) = 0
                {"Value in first bin (high)", 1, {true, 0}}, // floor(1*0.5) = 0
                {"Value exactly second edge", 2, {true, 1}}, // floor(2*0.5) = 1
                {"Value in second bin (high)", 3, {true, 1}}, // floor(3*0.5) = 1
                {"Value exactly third edge", 4, {true, 2}}, // floor(4*0.5) = 2
                {"Value in last bin (low)", 10, {true, 5}}, // floor(10*0.5) = 5
                {"Value in last bin (high)", 11, {true, 5}}, // floor(11*0.5) = 5
                {"Value exactly max", 12, {false, {}}}, // Should not bin
                {"Value above max", 13, {false, {}}},
            };
            runBinIdxTests(kernel, tests);
        }

        SECTION("Overflow Enabled, Negative Range [-10, -1], 9 bins")
        {
            // Total bins = 9 + 2 = 11. Bins: 0 (under), 1-9 (in range), 10 (over)
            // Edges: -10, -9, ..., -1
            // Bin width = (-1 - (-10)) / 9 = 9 / 9 = 1
            // scaling = static_cast<float>(9) / (-1 - (-10)) = 9.0f / 9.0f = 1.0f
            // Formula: floor((val - (-10)) * 1.0f) + 1 = floor(val + 10) + 1
            axis::AxisSplitting<int> split{{-10, -1}, 9, true};
            auto axis = axis::createLinear(split, createFunctorDescription<int>(identityFunctor, "test"));
            auto kernel = axis.getAxisKernel();

            std::vector<BinIdxTestCase<int>> tests = {
                {"Value below min", -11, {true, 0}},
                {"Value exactly min", -10, {true, 1}}, // floor(-10+10)+1 = 1
                {"Value in first bin", -10, {true, 1}}, // Re-test min
                {"Value exactly second edge", -9, {true, 2}}, // floor(-9+10)+1 = 2
                {"Value in second bin", -9, {true, 2}}, // Re-test edge
                {"Value in third bin", -8, {true, 3}}, // floor(-8+10)+1 = 3
                {"Value in last bin", -2, {true, 9}}, // floor(-2+10)+1 = 9
                {"Value exactly max", -1, {true, 10}}, // >= max goes to overflow
                {"Value above max", 0, {true, 10}},
                {"Value above max (positive)", 5, {true, 10}},
            };
            runBinIdxTests(kernel, tests);
        }
    }
}
