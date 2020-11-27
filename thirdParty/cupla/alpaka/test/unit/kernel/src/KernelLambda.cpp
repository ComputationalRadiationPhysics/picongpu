/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

// NVCC needs --expt-extended-lambda
#if !defined(__NVCC__) || (defined(__NVCC__) && defined(__CUDACC_EXTENDED_LAMBDA__))

#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/kernel/Traits.hpp>
#    include <alpaka/test/KernelExecutionFixture.hpp>
#    include <alpaka/test/acc/TestAccs.hpp>

#    include <catch2/catch.hpp>

//-----------------------------------------------------------------------------
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
        auto kernel = [] ALPAKA_FN_ACC(TAcc const& acc, bool* success) -> void {
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

//-----------------------------------------------------------------------------
struct TestTemplateArg
{
    template<typename TAcc>
    void operator()()
    {
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;

        alpaka::test::KernelExecutionFixture<TAcc> fixture(alpaka::Vec<Dim, Idx>::ones());

        std::uint32_t const arg = 42u;
        auto kernel = [] ALPAKA_FN_ACC(TAcc const& acc, bool* success, std::uint32_t const& arg1) -> void {
            alpaka::ignore_unused(acc);

            ALPAKA_CHECK(*success, 42u == arg1);
        };

        REQUIRE(fixture(kernel, arg));
    }
};

//-----------------------------------------------------------------------------
struct TestTemplateCapture
{
    template<typename TAcc>
    void operator()()
    {
        using Dim = alpaka::Dim<TAcc>;
        using Idx = alpaka::Idx<TAcc>;

        alpaka::test::KernelExecutionFixture<TAcc> fixture(alpaka::Vec<Dim, Idx>::ones());

        std::uint32_t const arg = 42u;

#    if BOOST_COMP_CLANG >= BOOST_VERSION_NUMBER(5, 0, 0)
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wunused-lambda-capture"
#    endif
        auto kernel = [arg] ALPAKA_FN_ACC(TAcc const& acc, bool* success) -> void {
            alpaka::ignore_unused(acc);

            ALPAKA_CHECK(*success, 42u == arg);
        };
#    if BOOST_COMP_CLANG >= BOOST_VERSION_NUMBER(5, 0, 0)
#        pragma clang diagnostic pop
#    endif

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
