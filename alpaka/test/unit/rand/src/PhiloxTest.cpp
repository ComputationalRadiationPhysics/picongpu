/* Copyright 2024 Jiri Vyskocil
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/rand/RandPhilox.hpp>
#include <alpaka/rand/RandPhiloxStateless.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

class PhiloxTest
{
protected:
    alpaka::rand::Philox4x32x10 statefulSingleEngine;
    alpaka::rand::Philox4x32x10Vector statefulVectorEngine;
};

TEST_CASE_METHOD(PhiloxTest, "HostStatefulVectorEngineTest")
{
    auto const resultVec = statefulVectorEngine();
    for(auto& result : resultVec)
    {
        REQUIRE(result >= statefulVectorEngine.min());
        REQUIRE(result <= statefulVectorEngine.max());
    }
}

TEST_CASE_METHOD(PhiloxTest, "HostStatefulSingleEngineTest")
{
    auto const result = statefulSingleEngine();
    REQUIRE(result >= statefulSingleEngine.min());
    REQUIRE(result <= statefulSingleEngine.max());
}

TEST_CASE("HostStatelessEngineTest")
{
    using Gen = alpaka::rand::PhiloxStateless4x32x10Vector;
    using Key = typename Gen::Key;
    using Counter = typename Gen::Counter;
    Key key = {42, 12345};
    Counter counter1 = {6789, 321, 0, 0};
    auto const result1 = Gen::generate(counter1, key);
    Counter counter2 = {6789, 321, 0, 1};
    auto const result2 = Gen::generate(counter2, key);
    // Make sure that the inputs are really expected to lead to different results.
    REQUIRE(result1 != result2);
}

template<typename T>
class PhiloxTestKernelSingle
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename T_Generator>
    ALPAKA_FN_ACC void genNumbers(TAcc const& acc, bool* success, T_Generator& gen) const
    {
        {
            static_cast<void>(acc);
            alpaka::rand::UniformReal<T> dist;
            auto const result = dist(gen);
            ALPAKA_CHECK(*success, static_cast<T>(0.0) <= result);
            ALPAKA_CHECK(*success, static_cast<T>(1.0) > result);
        }
    }

public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        // Philox generator for accelerator
        auto generator = alpaka::rand::Philox4x32x10(42, 12345, 6789);
        genNumbers<TAcc, decltype(generator)>(acc, success, generator);
    }
};

template<typename T>
class PhiloxTestKernelVector
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename T_Generator>
    ALPAKA_FN_ACC void genNumbers(TAcc const& acc, bool* success, T_Generator& gen) const
    {
        {
            static_cast<void>(acc);
            using DistributionResult = typename T_Generator::template ResultContainer<T>;
            alpaka::rand::UniformReal<DistributionResult> dist;
            auto const result = dist(gen);
            for(auto& element : result)
            {
                ALPAKA_CHECK(*success, static_cast<T>(0.0) <= element);
                ALPAKA_CHECK(*success, static_cast<T>(1.0) > element);
            }
        }
    }

public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        // Philox generator for accelerator
        auto generator = alpaka::rand::Philox4x32x10Vector(42, 12345, 6789);
        genNumbers<TAcc, decltype(generator)>(acc, success, generator);
    }
};

class PhiloxTestKernelStateless
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC void genNumbers(TAcc const& acc, bool* success) const
    {
        {
            static_cast<void>(acc);

            using Gen = alpaka::rand::PhiloxStateless4x32x10Vector;
            using Key = typename Gen::Key;
            using Counter = typename Gen::Counter;

            Key key = {42, 12345};
            Counter counter = {6789, 321, 0, 0};
            auto const result = Gen::generate(counter, key);

            size_t check = 0;
            for(auto& element : result)
            {
                check += element;
            }
            // Make sure the sequence is not in fact supposed to generate {0,0,0,0}.
            ALPAKA_CHECK(*success, check != 0);
        }
    }

public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        genNumbers<TAcc>(acc, success);
    }
};

TEMPLATE_LIST_TEST_CASE("PhiloxRandomGeneratorStatelessIsWorking", "[rand]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    PhiloxTestKernelStateless kernel;

    REQUIRE(fixture(kernel));
}

using TestScalars = std::tuple<float, double>;
using TestTypes = alpaka::meta::CartesianProduct<std::tuple, alpaka::test::TestAccs, TestScalars>;

TEMPLATE_LIST_TEST_CASE("PhiloxRandomGeneratorSingleIsWorking", "[rand]", TestTypes)
{
    using Acc = std::tuple_element_t<0, TestType>;
    using DataType = std::tuple_element_t<1, TestType>;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    PhiloxTestKernelSingle<DataType> kernel;

    REQUIRE(fixture(kernel));
}

TEMPLATE_LIST_TEST_CASE("PhiloxRandomGeneratorVectorIsWorking", "[rand]", TestTypes)
{
    using Acc = std::tuple_element_t<0, TestType>;
    using DataType = std::tuple_element_t<1, TestType>;
    using Idx = alpaka::Idx<Acc>;
    using Dim = alpaka::Dim<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    PhiloxTestKernelVector<DataType> kernel;

    REQUIRE(fixture(kernel));
}
