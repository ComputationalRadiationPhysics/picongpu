/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

// NVCC needs --expt-extended-lambda
#if !defined(__NVCC__) || (defined(__NVCC__) && defined(__CUDACC_EXTENDED_LAMBDA__))

#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/test/KernelExecutionFixture.hpp>
#    include <alpaka/test/acc/TestAccs.hpp>

#    include <catch2/catch_test_macros.hpp>

struct TestTemplateLambda
{
    template<typename TAcc>
    void operator()()
    {
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;

        alpaka::test::KernelExecutionFixture<TAcc> fixture(alpaka::Vec<Dim, Idx>::ones());

#    if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#        pragma warning(push)
#        pragma warning(disable : 4702) // warning C4702: unreachable code
#    endif
        auto kernel = [] ALPAKA_FN_ACC(TAcc const& acc, bool* success) -> void
        {
            ALPAKA_CHECK(
                *success,
                static_cast<alpaka::Idx<TAcc>>(1) == (alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)).prod());
        };
#    if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#        pragma warning(pop)
#    endif

        REQUIRE(fixture(kernel));
    }
};

struct TestTemplateArg
{
    template<typename TAcc>
    void operator()()
    {
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;

        alpaka::test::KernelExecutionFixture<TAcc> fixture(alpaka::Vec<Dim, Idx>::ones());

        std::uint32_t const arg = 42u;
        auto kernel = [] ALPAKA_FN_ACC(TAcc const& /* acc */, bool* success, std::uint32_t const& arg1) -> void
        { ALPAKA_CHECK(*success, 42u == arg1); };

        REQUIRE(fixture(kernel, arg));
    }
};

struct TestTemplateCapture
{
    template<typename TAcc>
    void operator()()
    {
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;

        alpaka::test::KernelExecutionFixture<TAcc> fixture(alpaka::Vec<Dim, Idx>::ones());

        std::uint32_t arg = 42u;

        auto kernel = [arg] ALPAKA_FN_ACC(TAcc const& /* acc */, bool* success) -> void
        { ALPAKA_CHECK(*success, 42u == arg); };

        REQUIRE(fixture(kernel));
    }
};

TEST_CASE("lambdaKernelIsWorking", "[kernel]")
{
    alpaka::meta::forEachType<alpaka::test::TestAccs>(TestTemplateLambda());
}

TEST_CASE("lambdaKernelWithArgumentIsWorking", "[kernel]")
{
    alpaka::meta::forEachType<alpaka::test::TestAccs>(TestTemplateArg());
}

TEST_CASE("lambdaKernelWithCapturingIsWorking", "[kernel]")
{
    alpaka::meta::forEachType<alpaka::test::TestAccs>(TestTemplateCapture());
}

#endif
