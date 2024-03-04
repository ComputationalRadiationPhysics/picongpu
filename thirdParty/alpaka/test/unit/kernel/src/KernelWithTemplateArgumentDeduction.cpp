/* Copyright 2020 Axel Huebl, Benjamin Worpitz, René Widera, Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/kernel/Traits.hpp>
#include <alpaka/meta/ForEachType.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

template<typename TExpected>
class KernelInvocationTemplateDeductionValueSemantics
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename Acc, typename TByValue, typename TByConstValue, typename TByConstReference>
    ALPAKA_FN_ACC auto operator()(
        Acc const& acc,
        bool* success,
        TByValue,
        TByConstValue const,
        TByConstReference const&) const -> void
    {
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::Idx<Acc>>(1) == (alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());

        static_assert(
            std::is_same_v<TByValue, TExpected>,
            "Incorrect first additional kernel template parameter type!");
        static_assert(
            std::is_same_v<TByConstValue, TExpected>,
            "Incorrect second additional kernel template parameter type!");
        static_assert(
            std::is_same_v<TByConstReference, TExpected>,
            "Incorrect third additional kernel template parameter type!");
    }
};

TEMPLATE_LIST_TEST_CASE("kernelFuntionObjectTemplateDeductionFromValue", "[kernel]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    using Value = std::int32_t;
    KernelInvocationTemplateDeductionValueSemantics<Value> kernel;

    Value value{};
    REQUIRE(fixture(kernel, value, value, value));
}

TEMPLATE_LIST_TEST_CASE("kernelFuntionObjectTemplateDeductionFromConstValue", "[kernel]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    using Value = std::int32_t;
    KernelInvocationTemplateDeductionValueSemantics<Value> kernel;

    Value const constValue{};
    REQUIRE(fixture(kernel, constValue, constValue, constValue));
}

TEMPLATE_LIST_TEST_CASE("kernelFuntionObjectTemplateDeductionFromConstReference", "[kernel]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    using Value = std::int32_t;
    KernelInvocationTemplateDeductionValueSemantics<Value> kernel;

    Value value{};
    Value const& constReference = value;
    REQUIRE(fixture(kernel, constReference, constReference, constReference));
}

template<typename TExpectedFirst, typename TExpectedSecond = TExpectedFirst>
class KernelInvocationTemplateDeductionPointerSemantics
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename Acc, typename TByPointer, typename TByPointerToConst>
    ALPAKA_FN_ACC auto operator()(Acc const& acc, bool* success, TByPointer*, TByPointerToConst const*) const -> void
    {
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::Idx<Acc>>(1) == (alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());

        static_assert(
            std::is_same_v<TByPointer, TExpectedFirst>,
            "Incorrect first additional kernel template parameter type!");
        static_assert(
            std::is_same_v<TByPointerToConst, TExpectedSecond>,
            "Incorrect second additional kernel template parameter type!");
    }
};

TEMPLATE_LIST_TEST_CASE("kernelFuntionObjectTemplateDeductionFromPointer", "[kernel]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    using Value = std::int32_t;
    KernelInvocationTemplateDeductionPointerSemantics<Value> kernel;

    Value value{};
    Value* pointer = &value;
    REQUIRE(fixture(kernel, pointer, pointer));
}

TEMPLATE_LIST_TEST_CASE("kernelFuntionObjectTemplateDeductionFromPointerToConst", "[kernel]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    using Value = std::int32_t;
    KernelInvocationTemplateDeductionPointerSemantics<Value const, Value> kernel;

    Value const constValue{};
    Value const* pointerToConst = &constValue;
    REQUIRE(fixture(kernel, pointerToConst, pointerToConst));
}

TEMPLATE_LIST_TEST_CASE("kernelFuntionObjectTemplateDeductionFromStaticArray", "[kernel]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    using Value = std::int32_t;
    KernelInvocationTemplateDeductionPointerSemantics<Value> kernel;

    Value staticArray[4] = {};
    REQUIRE(fixture(kernel, staticArray, staticArray));
}

TEMPLATE_LIST_TEST_CASE("kernelFuntionObjectTemplateDeductionFromConstStaticArray", "[kernel]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    using Value = std::int32_t;
    KernelInvocationTemplateDeductionPointerSemantics<Value const, Value> kernel;

    Value const constStaticArray[4] = {};
    REQUIRE(fixture(kernel, constStaticArray, constStaticArray));
}
