/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/rand/Traits.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch.hpp>

//#############################################################################
class RandTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename T_Generator>
    ALPAKA_FN_ACC void genNumbers(TAcc const& acc, bool* success, T_Generator& gen) const
    {
        {
            auto dist(alpaka::rand::distribution::createNormalReal<float>(acc));
            auto const r = dist(gen);
#if !BOOST_ARCH_PTX
            ALPAKA_CHECK(*success, std::isfinite(r));
#else
            alpaka::ignore_unused(r);
#endif
        }

        {
            auto dist(alpaka::rand::distribution::createNormalReal<double>(acc));
            auto const r = dist(gen);
#if !BOOST_ARCH_PTX
            ALPAKA_CHECK(*success, std::isfinite(r));
#else
            alpaka::ignore_unused(r);
#endif
        }
        {
            auto dist(alpaka::rand::distribution::createUniformReal<float>(acc));
            auto const r = dist(gen);
            ALPAKA_CHECK(*success, 0.0f <= r);
            ALPAKA_CHECK(*success, 1.0f > r);
        }

        {
            auto dist(alpaka::rand::distribution::createUniformReal<double>(acc));
            auto const r = dist(gen);
            ALPAKA_CHECK(*success, 0.0 <= r);
            ALPAKA_CHECK(*success, 1.0 > r);
        }

        {
            auto dist(alpaka::rand::distribution::createUniformUint<std::uint32_t>(acc));
            auto const r = dist(gen);
            alpaka::ignore_unused(r);
        }
    }

public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, bool* success) const -> void
    {
        // default generator for accelerator
        auto genDefault = alpaka::rand::generator::createDefault(acc, 12345u, 6789u);
        genNumbers(acc, success, genDefault);

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#    ifndef ALPAKA_ACC_ANY_BT_OMP5_ENABLED
        // TODO: These ifdefs are wrong: They will reduce the test to the
        // smallest common denominator from all enabled backends
        // std::random_device
        auto genRandomDevice = alpaka::rand::generator::createDefault(alpaka::rand::RandomDevice{}, 12345u, 6789u);
        genNumbers(acc, success, genRandomDevice);

        // MersenneTwister
        auto genMersenneTwister
            = alpaka::rand::generator::createDefault(alpaka::rand::MersenneTwister{}, 12345u, 6789u);
        genNumbers(acc, success, genMersenneTwister);
#    endif

        // TinyMersenneTwister
        auto genTinyMersenneTwister
            = alpaka::rand::generator::createDefault(alpaka::rand::TinyMersenneTwister{}, 12345u, 6789u);
        genNumbers(acc, success, genTinyMersenneTwister);
#endif
    }
};

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("defaultRandomGeneratorIsWorking", "[rand]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;

    alpaka::test::KernelExecutionFixture<Acc> fixture(alpaka::Vec<Dim, Idx>::ones());

    RandTestKernel kernel;

    REQUIRE(fixture(kernel));
}
