/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


// NVCC needs --expt-relaxed-constexpr
#if !defined(__NVCC__) || \
    ( defined(__NVCC__) && defined(__CUDACC_RELAXED_CONSTEXPR__) )

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/meta/ForEachType.hpp>

#include <catch2/catch.hpp>

#include <limits>

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
struct TestTemplate
{
template< typename TAcc >
void operator()()
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Idx = alpaka::idx::Idx<TAcc>;

    alpaka::test::KernelExecutionFixture<TAcc> fixture(
        alpaka::vec::Vec<Dim, Idx>::ones());

    KernelWithHostConstexpr kernel;

    REQUIRE(fixture(kernel));
}
};

TEST_CASE( "kernelWithHostConstexpr", "[kernel]")
{
    alpaka::meta::forEachType< alpaka::test::acc::TestAccs >( TestTemplate() );
}

#endif
