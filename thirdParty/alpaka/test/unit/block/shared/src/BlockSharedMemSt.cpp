/**
 * \file
 * Copyright 2015 Benjamin Worpitz
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
#include <alpaka/test/stream/Stream.hpp>
#include <alpaka/test/Array.hpp>
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

BOOST_AUTO_TEST_SUITE(blockSharedMemSt)

//#############################################################################
class BlockSharedMemStNonNullTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc) const
    -> void
    {
#if BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(6, 0, 0)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Waddress"  // warning: the compiler can assume that the address of ‘a’ will never be NULL [-Waddress]
#endif
        auto && a = alpaka::block::shared::st::allocVar<std::uint32_t, __COUNTER__>(acc);
        BOOST_VERIFY(static_cast<std::uint32_t *>(nullptr) != &a);

        auto && b = alpaka::block::shared::st::allocVar<std::uint32_t, __COUNTER__>(acc);
        BOOST_VERIFY(static_cast<std::uint32_t *>(nullptr) != &b);

        auto && c = alpaka::block::shared::st::allocVar<float, __COUNTER__>(acc);
        BOOST_VERIFY(static_cast<float *>(nullptr) != &c);

        auto && d = alpaka::block::shared::st::allocVar<double, __COUNTER__>(acc);
        BOOST_VERIFY(static_cast<double *>(nullptr) != &d);

        auto && e = alpaka::block::shared::st::allocVar<std::uint64_t, __COUNTER__>(acc);
        BOOST_VERIFY(static_cast<std::uint64_t *>(nullptr) != &e);


        auto && f = alpaka::block::shared::st::allocVar<alpaka::test::Array<std::uint32_t, 32>, __COUNTER__>(acc);
        BOOST_VERIFY(static_cast<std::uint32_t *>(nullptr) != &f[0]);

        auto && g = alpaka::block::shared::st::allocVar<alpaka::test::Array<std::uint32_t, 32>, __COUNTER__>(acc);
        BOOST_VERIFY(static_cast<std::uint32_t *>(nullptr) != &g[0]);

        auto && h = alpaka::block::shared::st::allocVar<alpaka::test::Array<double, 16>, __COUNTER__>(acc);
        BOOST_VERIFY(static_cast<double *>(nullptr) != &h[0]);
#if BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(6, 0, 0)
    #pragma GCC diagnostic pop
#endif
    }
};

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    nonNull,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Size>::ones());

    BlockSharedMemStNonNullTestKernel kernel;

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
}

//#############################################################################
class BlockSharedMemStSameTypeDifferentAdressTestKernel
{
public:
    //-----------------------------------------------------------------------------
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc) const
    -> void
    {
        auto && a = alpaka::block::shared::st::allocVar<std::uint32_t, __COUNTER__>(acc);
        auto && b = alpaka::block::shared::st::allocVar<std::uint32_t, __COUNTER__>(acc);
        BOOST_VERIFY(&a != &b);
        auto && c = alpaka::block::shared::st::allocVar<std::uint32_t, __COUNTER__>(acc);
        BOOST_VERIFY(&b != &c);
        BOOST_VERIFY(&a != &c);
        BOOST_VERIFY(&b != &c);

        auto && d = alpaka::block::shared::st::allocVar<alpaka::test::Array<std::uint32_t, 32>, __COUNTER__>(acc);
        BOOST_VERIFY(&a != &d[0]);
        BOOST_VERIFY(&b != &d[0]);
        BOOST_VERIFY(&c != &d[0]);
        auto && e = alpaka::block::shared::st::allocVar<alpaka::test::Array<std::uint32_t, 32>, __COUNTER__>(acc);
        BOOST_VERIFY(&a != &e[0]);
        BOOST_VERIFY(&b != &e[0]);
        BOOST_VERIFY(&c != &e[0]);
        BOOST_VERIFY(&d[0] != &e[0]);
    }
};

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    sameTypeDifferentAdress,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Size>::ones());

    BlockSharedMemStSameTypeDifferentAdressTestKernel kernel;

    BOOST_REQUIRE_EQUAL(
        true,
        fixture(
            kernel));
}

BOOST_AUTO_TEST_SUITE_END()
