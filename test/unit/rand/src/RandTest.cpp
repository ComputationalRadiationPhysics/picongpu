/**
 * \file
 * Copyright 2017-2018 Benjamin Worpitz, Axel Huebl
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

// \Hack: Boost.MPL defines BOOST_MPL_CFG_GPU_ENABLED to __host__ __device__ if nvcc is used.
// BOOST_AUTO_TEST_CASE_TEMPLATE and its internals are not GPU enabled but is using boost::mpl::for_each internally.
// For each template parameter this leads to:
// /home/travis/build/boost/boost/mpl/for_each.hpp(78): warning: calling a __host__ function from a __host__ __device__ function is not allowed
// because boost::mpl::for_each has the BOOST_MPL_CFG_GPU_ENABLED attribute but the test internals are pure host methods.
// Because we do not use MPL within GPU code here, we can disable the MPL GPU support.
#define BOOST_MPL_CFG_GPU_ENABLED

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <alpaka/core/BoostPredef.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

BOOST_AUTO_TEST_SUITE(rand_)

//#############################################################################
class RandTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename T_Generator
    >
    ALPAKA_FN_ACC void
    genNumbers(
        TAcc const & acc,
        bool * success,
        T_Generator & gen
    ) const
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
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool * success) const
    -> void
    {
        // default generator for accelerator
        auto genDefault = alpaka::rand::generator::createDefault(
            acc,
            12345u,
            6789u
        );
        genNumbers( acc, success, genDefault );

#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && \
  !defined(ALPAKA_ACC_GPU_HIP_ENABLED)
        // std::random_device
        auto genRandomDevice = alpaka::rand::generator::createDefault(
            alpaka::rand::RandomDevice{},
            12345u,
            6789u
        );
        genNumbers( acc, success, genRandomDevice );

        // MersenneTwister
        auto genMersenneTwister = alpaka::rand::generator::createDefault(
            alpaka::rand::MersenneTwister{},
            12345u,
            6789u
        );
        genNumbers( acc, success, genMersenneTwister );

        // TinyMersenneTwister
        auto genTinyMersenneTwister = alpaka::rand::generator::createDefault(
            alpaka::rand::TinyMersenneTwister{},
            12345u,
            6789u
        );
        genNumbers( acc, success, genTinyMersenneTwister );
#endif
    }
};

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    defaultRandomGeneratorIsWorking,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    RandTestKernel kernel;

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
}

BOOST_AUTO_TEST_SUITE_END()
