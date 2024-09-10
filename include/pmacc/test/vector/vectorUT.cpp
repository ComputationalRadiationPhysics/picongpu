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
#include <pmacc/algorithms/TypeCast.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/test/PMaccFixture.hpp>
#include <pmacc/verify.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>

#include <catch2/catch_test_macros.hpp>

#include <pmacc/math/vector/compile-time/UInt32.hpp>

/** @file
 *
 *  This file is testing vector functionality
 */

using MyPMaccFixture = pmacc::test::PMaccFixture<TEST_DIM>;
static MyPMaccFixture fixture;

TEST_CASE("vector constructor generator", "[vector]")
{
    using namespace pmacc;
    using namespace pmacc::math;
    uint32_t const numElements = 2u;
    using VecType = Vector<uint32_t, TEST_DIM>;

    auto hostDeviceBuffer = HostDeviceBuffer<VecType, DIM1>(DataSpace<DIM1>{numElements});
    using DeviceBuf = DeviceBuffer<uint32_t, DIM1>;

    hostDeviceBuffer.getDeviceBuffer().setValue(VecType::create(42));
    auto const testKernel = [] ALPAKA_FN_ACC(auto const& acc, VecType* data)
    {
        // constexpr lambda generator
        constexpr auto vec = Vector<uint32_t, TEST_DIM>([](uint32_t const i) constexpr { return i; });
        data[0] = vec;
        // non constexpr lambda generator
        data[1] = Vector<uint32_t, TEST_DIM>([](uint32_t const i) { return i * 2u; });
    };
    PMACC_KERNEL(testKernel)(1, 1)(hostDeviceBuffer.getDeviceBuffer().data());
    hostDeviceBuffer.deviceToHost();

    REQUIRE(hostDeviceBuffer.getHostBuffer().data()[0] == Vector<uint32_t, 3u>(0u, 1u, 2u).shrink<TEST_DIM>());
    REQUIRE(hostDeviceBuffer.getHostBuffer().data()[1] == Vector<uint32_t, 3u>(0u, 2u, 4u).shrink<TEST_DIM>());
}

/** define one dimensional vector compile time test cases for operator +,-,*,/ */
struct CompileTimeKernel1D
{
    template<typename T_Acc>
    DINLINE void operator()(T_Acc const&) const
    {
        using namespace pmacc::math;

        constexpr auto vec = Vector{3};
        static_assert(vec.size() == 1);
        static_assert(vec.x() == 3);
        static_assert(vec == Vector{3});

        constexpr auto typeLambda = [](auto const typeDummy) constexpr
        {
            using type = std::decay_t<decltype(typeDummy)>;

            constexpr auto inputData = std::make_tuple(
                std::make_tuple(std::plus{}, Vector(type{3}), Vector(type{7}), Vector(type{10})),
                std::make_tuple(std::plus{}, Vector(type{3}), type{7}, Vector(type{10})),
                std::make_tuple(std::plus{}, type{3}, Vector(type{7}), Vector(type{10})),

                std::make_tuple(std::minus{}, Vector(type{17}), Vector(type{7}), Vector(type{10})),
                std::make_tuple(std::minus{}, Vector(type{17}), type{7}, Vector(type{10})),
                std::make_tuple(std::minus{}, type{17}, Vector(type{7}), Vector(type{10})),

                std::make_tuple(std::multiplies{}, Vector(type{3}), Vector(type{7}), Vector(type{21})),
                std::make_tuple(std::multiplies{}, Vector(type{3}), type{7}, Vector(type{21})),
                std::make_tuple(std::multiplies{}, type{3}, Vector(type{7}), Vector(type{21})),

                std::make_tuple(std::divides{}, Vector(type{21}), Vector(type{7}), Vector(type{3})),
                std::make_tuple(std::divides{}, Vector(type{21}), type{7}, Vector(type{3})),
                std::make_tuple(std::divides{}, type{21}, Vector(type{7}), Vector(type{3})));
            constexpr bool x = std::apply(
                [&](auto... args) constexpr
                { return ((std::get<0>(args)(std::get<1>(args), std::get<2>(args)) == std::get<3>(args)) && ...); },
                inputData);
            return x;
        };

        constexpr auto inputTypes = std::tuple<int, uint32_t, uint64_t, float, double>{};
        constexpr bool x = std::apply(
            [&](auto... args) constexpr { return (typeLambda(args) && ...); },
            inputTypes);
        static_assert(x);
    }
};

/** define two dimensional vector compile time test cases for operator +,-,*,/ */
struct CompileTimeKernel2D
{
    template<typename T_Acc>
    DINLINE void operator()(T_Acc const&) const
    {
        using namespace pmacc::math;

        constexpr auto vec = Vector{3, 7};
        static_assert(vec.size() == 2);
        static_assert(vec.x() == 3 && vec.y() == 7);
        static_assert(vec == Vector{3, 7});
        static_assert(vec != Vector{7, 3});

        constexpr auto typeLambda = [](auto const typeDummy) constexpr
        {
            using type = std::decay_t<decltype(typeDummy)>;

            constexpr auto inputData = std::make_tuple(
                std::make_tuple(
                    std::plus{},
                    Vector(type{3}, type{7}),
                    Vector(type{7}, type{9}),
                    Vector(type{10}, type{16})),
                std::make_tuple(std::plus{}, Vector(type{3}, type{9}), type{7}, Vector(type{10}, type{16})),
                std::make_tuple(std::plus{}, type{3}, Vector(type{7}, type{9}), Vector(type{10}, type{12})),

                std::make_tuple(
                    std::minus{},
                    Vector(type{17}, type{7}),
                    Vector(type{7}, type{3}),
                    Vector(type{10}, type{4})),
                std::make_tuple(std::minus{}, Vector(type{17}, type{7}), type{7}, Vector(type{10}, type{0})),
                std::make_tuple(std::minus{}, type{17}, Vector(type{7}, type{3}), Vector(type{10}, type{14})),

                std::make_tuple(
                    std::multiplies{},
                    Vector(type{3}, type{7}),
                    Vector(type{7}, type{11}),
                    Vector(type{21}, type{77})),
                std::make_tuple(std::multiplies{}, Vector(type{3}, type{7}), type{7}, Vector(type{21}, type{49})),
                std::make_tuple(std::multiplies{}, type{3}, Vector(type{7}, type{3}), Vector(type{21}, type{9})),

                std::make_tuple(
                    std::divides{},
                    Vector(type{21}, type{3}),
                    Vector(type{7}, type{3}),
                    Vector(type{3}, type{1})),
                std::make_tuple(std::divides{}, Vector(type{21}, type{14}), type{7}, Vector(type{3}, type{2})),
                std::make_tuple(std::divides{}, type{21}, Vector(type{7}, type{3}), Vector(type{3}, type{7})));
            constexpr bool x = std::apply(
                [&](auto... args) constexpr
                { return ((std::get<0>(args)(std::get<1>(args), std::get<2>(args)) == std::get<3>(args)) && ...); },
                inputData);
            return x;
        };

        constexpr auto inputTypes = std::tuple<int, uint32_t, uint64_t, float, double>{};
        constexpr bool x = std::apply(
            [&](auto... args) constexpr { return (typeLambda(args) && ...); },
            inputTypes);
        static_assert(x);
    }
};

/** define two dimensional vector compile time test cases for operator >,>=,<,<= */
struct CompileTimeKernelCompare2D
{
    template<typename T_Acc>
    DINLINE void operator()(T_Acc const&) const
    {
        using namespace pmacc::math;

        constexpr auto typeLambda = [](auto const typeDummy) constexpr
        {
            using type = std::decay_t<decltype(typeDummy)>;

            constexpr auto inputData = std::make_tuple(
                std::make_tuple(
                    std::greater{},
                    Vector(type{3}, type{7}),
                    Vector(type{7}, type{9}),
                    Vector(false, false)),
                std::make_tuple(std::greater{}, Vector(type{3}, type{9}), type{7}, Vector(false, true)),
                std::make_tuple(std::greater{}, type{3}, Vector(type{7}, type{9}), Vector(false, false)),

                std::make_tuple(
                    std::greater_equal{},
                    Vector(type{3}, type{7}),
                    Vector(type{3}, type{9}),
                    Vector(true, false)),
                std::make_tuple(std::greater_equal{}, Vector(type{3}, type{9}), type{3}, Vector(true, true)),
                std::make_tuple(std::greater_equal{}, type{3}, Vector(type{7}, type{9}), Vector(false, false)),

                std::make_tuple(std::less{}, Vector(type{3}, type{7}), Vector(type{7}, type{9}), Vector(true, true)),
                std::make_tuple(std::less{}, Vector(type{3}, type{9}), type{7}, Vector(true, false)),
                std::make_tuple(std::less{}, type{3}, Vector(type{7}, type{9}), Vector(true, true)),

                std::make_tuple(
                    std::less_equal{},
                    Vector(type{3}, type{7}),
                    Vector(type{3}, type{9}),
                    Vector(true, true)),
                std::make_tuple(std::less_equal{}, Vector(type{3}, type{9}), type{3}, Vector(true, false)),
                std::make_tuple(std::less_equal{}, type{3}, Vector(type{7}, type{9}), Vector(true, true))

            );
            constexpr bool x = std::apply(
                [&](auto... args) constexpr
                { return ((std::get<0>(args)(std::get<1>(args), std::get<2>(args)) == std::get<3>(args)) && ...); },
                inputData);
            return x;
        };

        constexpr auto inputTypes = std::tuple<int, uint32_t, uint64_t, float, double>{};
        constexpr bool x = std::apply(
            [&](auto... args) constexpr { return (typeLambda(args) && ...); },
            inputTypes);
        static_assert(x);
    }
};

PMACC_CONST_VECTOR(float, 3u, cellSize, 1.0, 2.0, 3.0);

/** define N vector compile/runtime time tests for different operations */
struct RunTimeKernel
{
    template<typename T_Acc>
    DINLINE void operator()(T_Acc const&, bool* data, size_t* numTestsOut) const
    {
        using namespace pmacc::math;
        size_t i = 0u;
        // DIM1
        {
            // test template type deduction
            auto vec = Vector(42u);
            static_assert(vec.size() == 1u);
            data[i++] = vec.x() == 42;
        }
        // DIM2
        {
            // test template type deduction
            auto vec = Vector(42u, 43u);
            static_assert(vec.size() == 2u);
            data[i++] = vec.x() == 42u && vec.y() == 43u;
        }
        {
            // revert vector
            constexpr auto vec = Vector{42, 43};
            constexpr auto vecRevert = vec.revert();
            static_assert(vecRevert.size() == 2u);
            data[i++] = vecRevert.x() == 43u && vecRevert.y() == 42u;
        }
        {
            // explicit cast type
            constexpr auto vec = Vector<uint32_t, 2u>(42, 43);
            Vector<uint64_t, 2u> vecCopy(vec);
            static_assert(vecCopy.size() == 2u);
            data[i++] = vecCopy.x() == 42u && vecCopy.y() == 43u;
        }
        {
            constexpr auto vec = Vector<uint32_t, 2u>{42, 43};
            constexpr auto vecShrinked = vec.shrink<1u>();
            static_assert(vecShrinked.size() == 1u);
            data[i++] = vecShrinked.x() == 42u;
        }
        {
            constexpr auto vec = Vector<uint32_t, 2u>{42, 43};
            constexpr auto vecRemoved = vec.remove<1u>();
            static_assert(vecRemoved.size() == 1u);
            data[i++] = vecRemoved.x() == 42u;
        }
        {
            constexpr auto vec1 = Vector<uint32_t, 2u>(3, 5);
            constexpr auto vec2 = Vector<uint32_t, 2u>(7, 11);
            constexpr auto vecRes = vec1 + vec2;
            constexpr bool res = vecRes == Vector(10u, 16u);
            data[i++] = res;
        }
        {
            // unary assign operation
            auto vec = Vector<uint32_t, 2u>(42, 43);
            vec += 1;
            static_assert(vec.size() == 2u);
            data[i] = vec.x() == 43u && vec.y() == 44u;
            vec -= 1;
            data[i] &= vec.x() == 42u && vec.y() == 43u;
            vec *= 2;
            data[i] &= vec.x() == 84u && vec.y() == 86u;
            vec /= 2;
            data[i++] &= vec.x() == 42u && vec.y() == 43u;
        }
        // DIM3
        {
            auto vec = Vector<uint32_t, 3u>(42, 43, 44);
            static_assert(vec.size() == 3u);
            data[i++] = vec.x() == 42u && vec.y() == 43u && vec.z() == 44u;
        }
        {
            constexpr auto vec = Vector<uint32_t, 3u>::create(42);
            static_assert(vec.size() == 3u);
            data[i++] = vec.x() == vec.y() && vec.y() == vec.z() && vec.z() == 42u;
        }
        {
            auto vec = Vector<uint32_t, 3u>::create(42);
            static_assert(vec.size() == 3u);
            data[i++] = vec.x() == vec.y() && vec.y() == vec.z() && vec.z() == 42u;
        }
        {
            constexpr auto vec = Vector<uint32_t, 3u>::create(1);
            static_assert(vec.size() == vec.sumOfComponents());
            constexpr auto vec2 = Vector<uint32_t, 3u>(2u, 3u, 4u);
            data[i] = vec2.sumOfComponents() == 10u;
            data[i++] = vec2.productOfComponents() == 24u;
        }
        {
            // negation
            constexpr auto vec = Vector<int, 3u>::create(1);
            constexpr auto vec2 = -vec;
            static_assert(vec2 == Vector<int, 3u>::create(-1));
            auto vecRuntime = Vector<int, 3u>::create(1);
            auto vecRuntime2 = -vecRuntime;
            data[i++] = vecRuntime2 == Vector<int, 3u>::create(-1);
        }
        {
            // modulo
            constexpr auto vec = Vector{4, 3, 2};
            constexpr auto vec2 = vec % 2;
            static_assert(vec2 == Vector<int, 3u>{0, 1, 0});
            auto vecRuntime = Vector(4, 3, 2);
            auto vecRuntime2 = vecRuntime % 3;
            data[i++] = vecRuntime2 == Vector<int, 3u>(1, 0, 2);
        }
        {
            // value type cast
            auto vec = Vector(4, 3, 2);
            auto vec2 = Vector<float, 3>(vec);
            auto vec3 = pmacc::algorithms::precisionCast::precisionCast<float>(vec);
            auto vec4 = pmacc::algorithms::precisionCast::precisionCast<double>(cellSize);
            // cast to same type the vector already contains
            auto vec5 = pmacc::algorithms::precisionCast::precisionCast<float>(cellSize);
            data[i] = vec2 == Vector<float, 3u>(4.f, 3.f, 2.f);
            data[i] = vec3 == Vector<float, 3u>(4.f, 3.f, 2.f);
            data[i] = vec4 == Vector<double, 3u>(1., 2., 3.);
            data[i++] = vec5 == Vector<float, 3u>(1., 2., 3.);
        }
        {
            // assignment to default storage policy
            Vector<float, 3u> vec = cellSize;
            data[i++] = vec == Vector<float, 3u>(1., 2., 3.);
        }
        {
            // precisionCast
            constexpr auto vec = Vector<int, 3u>{4, 3, 2};
            constexpr auto vec2 = pmacc::algorithms::precisionCast::precisionCast<float>(vec);
            static_assert(vec2 == Vector<float, 3u>{4.f, 3.f, 2.f});
            data[i++] = vec2 == Vector<float, 3u>{4.f, 3.f, 2.f};
        }

        *numTestsOut = i;
    }
};

TEST_CASE("vector generic", "[vector]")
{
    using namespace pmacc;
    using namespace pmacc::math;

    // increase this variable in case new tests get added to RunTimeKernel
    size_t const numElements = 17u;

    auto hostDeviceBuffer = HostDeviceBuffer<bool, DIM1>(DataSpace<DIM1>{numElements});
    auto numTestsBuffer = HostDeviceBuffer<size_t, DIM1>(DataSpace<DIM1>{1});
    numTestsBuffer.getDeviceBuffer().setValue(0u);
    using DeviceBuf = DeviceBuffer<uint32_t, DIM1>;

    hostDeviceBuffer.getDeviceBuffer().setValue(false);

    PMACC_KERNEL(CompileTimeKernel1D{})(1, 1)();
    PMACC_KERNEL(CompileTimeKernel2D{})(1, 1)();
    PMACC_KERNEL(CompileTimeKernelCompare2D{})(1, 1)();

    PMACC_KERNEL(RunTimeKernel{})
    (1, 1)(hostDeviceBuffer.getDeviceBuffer().data(), numTestsBuffer.getDeviceBuffer().data());
    hostDeviceBuffer.deviceToHost();
    numTestsBuffer.deviceToHost();
    // check that all tests got executed and that the array is not too small.
    REQUIRE(numTestsBuffer.getHostBuffer().data()[0] == numElements);

    for(size_t i = 0; i < hostDeviceBuffer.getHostBuffer().size(); ++i)
    {
        bool res = hostDeviceBuffer.getHostBuffer().data()[i] == true;

        if(!res)
        {
            std::cerr << "test: " << i << " FAILED." << std::endl;
        }
        REQUIRE(res == true);
    }
}
