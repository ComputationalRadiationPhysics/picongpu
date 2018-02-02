/**
 * \file
 * Copyright 2017 Benjamin Worpitz
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

#include <boost/assert.hpp>
#include <boost/predef.h>
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
public:
#if BOOST_COMP_GNUC
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-variable"
#endif
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc) const
    -> void
    {
        auto gen(alpaka::rand::generator::createDefault(acc, 12345u, 6789u));

#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-variable"
#endif
// gcc 5.4 in combination with nvcc 8.0 fails to compile the CPU STL distributions when --expt-relaxed-constexpr is enabled
// /usr/include/c++/5/cmath(362): error: calling a __host__ function("__builtin_logl") from a __device__ function("std::log") is not allowed
#if !((BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(5, 0, 0)) && (BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(8, 0, 0)) && (BOOST_COMP_NVCC < BOOST_VERSION_NUMBER(9, 0, 0)))
        {
            auto dist(alpaka::rand::distribution::createNormalReal<float>(acc));
            auto const r = dist(gen);
#if !BOOST_ARCH_CUDA_DEVICE
            BOOST_VERIFY(std::isfinite(r));
#endif
        }

        {
            auto dist(alpaka::rand::distribution::createNormalReal<double>(acc));
            auto const r = dist(gen);
#if !BOOST_ARCH_CUDA_DEVICE
            BOOST_VERIFY(std::isfinite(r));
#endif
        }
        {
            auto dist(alpaka::rand::distribution::createUniformReal<float>(acc));
            auto const r = dist(gen);
            BOOST_VERIFY(0.0f <= r);
            BOOST_VERIFY(1.0f > r);
        }

        {
            auto dist(alpaka::rand::distribution::createUniformReal<double>(acc));
            auto const r = dist(gen);
            BOOST_VERIFY(0.0 <= r);
            BOOST_VERIFY(1.0 > r);
        }

        {
            auto dist(alpaka::rand::distribution::createUniformUint<std::uint32_t>(acc));
            auto const r = dist(gen);
        }
#endif
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif
    }
#if BOOST_COMP_GNUC
    #pragma GCC diagnostic pop
#endif
};

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    defaultRandomGeneratorIsWorking,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Size>::ones());

    RandTestKernel kernel;

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
}

BOOST_AUTO_TEST_SUITE_END()
