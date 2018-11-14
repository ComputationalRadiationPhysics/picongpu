/**
 * \file
 * Copyright 2017 Rene Widera, Benjamin Worpitz
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

// NVCC needs --expt-relaxed-constexpr
#if !defined(__NVCC__) || \
    ( defined(__NVCC__) && defined(__CUDACC_RELAXED_CONSTEXPR__) )

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

#include <limits>

BOOST_AUTO_TEST_SUITE(kernel)

//#############################################################################
//!
//#############################################################################
class KernelWithHostConstexpr
{
public:
    //-----------------------------------------------------------------------------
    //!
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        bool* success) const
    -> void
    {
        alpaka::ignore_unused(acc);

#if BOOST_COMP_MSVC
    #pragma warning(push)
    #pragma warning(disable: 4127)  // warning C4127: conditional expression is constant
#endif
        // FIXME: workaround for HIP(HCC) where numeric_limits::* do not provide
        // matching host-device restriction requirements
#if defined(BOOST_COMP_HCC) && BOOST_COMP_HCC
        constexpr auto max = static_cast<std::uint32_t>(-1);
#else
        constexpr auto max = std::numeric_limits< std::uint32_t >::max();
#endif
        ALPAKA_CHECK(*success, 0 != max);
#if BOOST_COMP_MSVC
    #pragma warning(pop)
#endif
    }
};

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    kernelWithHostConstexpr,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    KernelWithHostConstexpr kernel;

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
}

BOOST_AUTO_TEST_SUITE_END()

#endif
