/* Copyright 2022 Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

// For nvcc these tests require extended lambda.
// Additionally disable the tests for Visual Studio + CUDA since it causes the following compile error:
// "On Windows, the enclosing parent function ("C_A_T_C_H_T_E_M_P_L_A_T_E_T_E_S_T_F_U_N_C_...") for an extended
// __host__
// __device__ lambda cannot have internal or no linkage"
#if(!defined(__NVCC__) || (defined(__NVCC__) && defined(__CUDACC_EXTENDED_LAMBDA__) && !defined(_MSC_VER)))

#    include "Functor.hpp"
#    include "TestTemplate.hpp"

#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/math/Complex.hpp>
#    include <alpaka/test/KernelExecutionFixture.hpp>
#    include <alpaka/test/acc/TestAccs.hpp>

#    include <catch2/catch_template_test_macros.hpp>

#    include <cstdint>
#    include <tuple>

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::size_t>;

//! Caller of test template with additional lambda wrapping the functor
//! @tparam TAcc Accelerator.
//! @tparam TData Input data type..
template<typename TAcc, typename TData>
struct LambdaMathTestTemplate
{
    //! Run the test for one of the given functors
    //! @tparam TFunctors Typelist of functors, each defined in Functor.hpp.
    template<typename TFunctors>
    void operator()() const
    {
        // To save on compiler memory and run time we test for only one of the functors and not all of them.
        // The behavior should not differ between those as we already check all the functors without lambda.
        // To avoid always using the first functor from the list, calculate index based on input data type sizes.
        constexpr uint32_t index = (sizeof(TAcc) + sizeof(TData) + sizeof(TFunctors)) % std::tuple_size_v<TFunctors>;
        using Functor = std::tuple_element_t<index, TFunctors>;
        using ArgsItem = mathtest::ArgsItem<TData, Functor::arity>;
        auto wrappedFunctor
            = [] ALPAKA_FN_HOST_ACC(ArgsItem const& arguments, TAcc const& acc) { return Functor{}(arguments, acc); };
        auto testTemplate = mathtest::TestTemplate<TAcc, Functor>{};
        testTemplate.template operator()<TData>(wrappedFunctor);
    }
};

TEMPLATE_LIST_TEST_CASE("mathOpsLambdaFloat", "[math] [operator]", TestAccs)
{
    using Acc = TestType;
    auto testTemplate = LambdaMathTestTemplate<Acc, float>{};
    testTemplate.template operator()<mathtest::UnaryFunctorsReal>();
    testTemplate.template operator()<mathtest::BinaryFunctorsReal>();
    testTemplate.template operator()<mathtest::TernaryFunctorsReal>();
}

TEMPLATE_LIST_TEST_CASE("mathOpsLambdaDouble", "[math] [operator]", TestAccs)
{
    using Acc = TestType;
    auto testTemplate = LambdaMathTestTemplate<Acc, double>{};
    testTemplate.template operator()<mathtest::UnaryFunctorsReal>();
    testTemplate.template operator()<mathtest::BinaryFunctorsReal>();
    testTemplate.template operator()<mathtest::TernaryFunctorsReal>();
}

TEMPLATE_LIST_TEST_CASE("mathOpsLambdaComplexFloat", "[math] [operator]", TestAccs)
{
    using Acc = TestType;
    auto testTemplate = LambdaMathTestTemplate<Acc, alpaka::Complex<float>>{};
    testTemplate.template operator()<mathtest::UnaryFunctorsComplex>();
    testTemplate.template operator()<mathtest::BinaryFunctorsComplex>();
}

TEMPLATE_LIST_TEST_CASE("mathOpsLambdaComplexDouble", "[math] [operator]", TestAccs)
{
    using Acc = TestType;
    auto testTemplate = LambdaMathTestTemplate<Acc, alpaka::Complex<double>>{};
    testTemplate.template operator()<mathtest::UnaryFunctorsComplex>();
    testTemplate.template operator()<mathtest::BinaryFunctorsComplex>();
}
#endif
