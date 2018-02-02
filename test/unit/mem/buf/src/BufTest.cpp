/**
 * \file
 * Copyright 2015-2017 Benjamin Worpitz
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
#include <alpaka/test/mem/view/ViewTest.hpp>

#include <boost/predef.h>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

#include <type_traits>
#include <numeric>

BOOST_AUTO_TEST_SUITE(memBuf)

//#############################################################################
//! 1D: sizeof(TSize) * (5)
//! 2D: sizeof(TSize) * (5, 4)
//! 3D: sizeof(TSize) * (5, 4, 3)
//! 4D: sizeof(TSize) * (5, 4, 3, 2)
template<
    std::size_t Tidx>
struct CreateExtentBufVal
{
    //-----------------------------------------------------------------------------
    template<
        typename TSize>
    static auto create(
        TSize)
    -> TSize
    {
        return sizeof(TSize) * (5u - Tidx);
    }
};

//-----------------------------------------------------------------------------
template<
    typename TAcc>
static auto basicBufferOperationsTest(
    alpaka::vec::Vec<alpaka::dim::Dim<TAcc>, alpaka::size::Size<TAcc>> const & extent)
-> void
{
    using Dev = alpaka::dev::Dev<TAcc>;
    using Pltf = alpaka::pltf::Pltf<Dev>;
    using Stream = alpaka::test::stream::DefaultStream<Dev>;

    using Elem = float;
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;

    Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));
    Stream stream(dev);

    //-----------------------------------------------------------------------------
    // alpaka::mem::buf::alloc
    auto buf(alpaka::mem::buf::alloc<Elem, Size>(dev, extent));

    //-----------------------------------------------------------------------------
    auto const offset(alpaka::vec::Vec<Dim, Size>::zeros());
    alpaka::test::mem::view::viewTestImmutable<
        Elem>(
            buf,
            dev,
            extent,
            offset);

    //-----------------------------------------------------------------------------
    alpaka::test::mem::view::viewTestMutable<
        TAcc>(
            stream,
            buf);
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    memBufBasicTest,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;

    auto const extent(alpaka::vec::createVecFromIndexedFnWorkaround<Dim, Size, CreateExtentBufVal>(Size()));

    basicBufferOperationsTest<
        TAcc>(
            extent);
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    memBufZeroSizeTest,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;

    auto const extent(alpaka::vec::Vec<Dim, Size>::zeros());

    basicBufferOperationsTest<
        TAcc>(
            extent);
}

BOOST_AUTO_TEST_SUITE_END()
