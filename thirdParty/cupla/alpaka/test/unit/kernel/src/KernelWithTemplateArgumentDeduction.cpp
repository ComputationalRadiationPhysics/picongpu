/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera, Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/kernel/Traits.hpp>
#include <alpaka/meta/ForEachType.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch.hpp>

#include <type_traits>

//#############################################################################
template<typename TExpected>
class KernelInvocationTemplateDeductionValueSemantics
{
public:
    //-----------------------------------------------------------------------------
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
            std::is_same<TByValue, TExpected>::value,
            "Incorrect first additional kernel template parameter type!");
        static_assert(
            std::is_same<TByConstValue, TExpected>::value,
            "Incorrect second additional kernel template parameter type!");
        static_assert(
            std::is_same<TByConstReference, TExpected>::value,
            "Incorrect third additional kernel template parameter type!");
    }
};

//-----------------------------------------------------------------------------
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

//#############################################################################
template<typename TExpectedFirst, typename TExpectedSecond = TExpectedFirst>
class KernelInvocationTemplateDeductionPointerSemantics
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename Acc, typename TByPointer, typename TByPointerToConst>
    ALPAKA_FN_ACC auto operator()(Acc const& acc, bool* success, TByPointer*, TByPointerToConst const*) const -> void
    {
        ALPAKA_CHECK(
            *success,
            static_cast<alpaka::Idx<Acc>>(1) == (alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());

        static_assert(
            std::is_same<TByPointer, TExpectedFirst>::value,
            "Incorrect first additional kernel template parameter type!");
        static_assert(
            std::is_same<TByPointerToConst, TExpectedSecond>::value,
            "Incorrect second additional kernel template parameter type!");
    }
};

//-----------------------------------------------------------------------------
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
