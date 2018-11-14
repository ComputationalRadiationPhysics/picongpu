/**
 * \file
 * Copyright 2018 Benjamin Worpitz
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
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/Extent.hpp>

#include <alpaka/core/BoostPredef.hpp>
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


#if BOOST_COMP_GNUC
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wcast-align" // "cast from 'std::uint8_t*' to 'Elem*' increases required alignment of target type"
#endif

namespace alpaka
{
namespace test
{
namespace mem
{
namespace view
{
    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    auto testViewPlainPtrImmutable(
        alpaka::mem::view::ViewPlainPtr<TDev, TElem, TDim, TIdx> const & view,
        TDev const & dev,
        alpaka::vec::Vec<TDim, TIdx> const & extentView,
        alpaka::vec::Vec<TDim, TIdx> const & offsetView)
    -> void
    {
        //-----------------------------------------------------------------------------
        alpaka::test::mem::view::testViewImmutable<
            TElem>(
                view,
                dev,
                extentView,
                offsetView);
    }

    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx>
    auto testViewPlainPtrMutable(
        alpaka::mem::view::ViewPlainPtr<TDev, TElem, TDim, TIdx> & view,
        TDev const & dev,
        alpaka::vec::Vec<TDim, TIdx> const & extentView,
        alpaka::vec::Vec<TDim, TIdx> const & offsetView)
    -> void
    {
        //-----------------------------------------------------------------------------
        testViewPlainPtrImmutable<
            TAcc>(
                view,
                dev,
                extentView,
                offsetView);

        using Queue = alpaka::test::queue::DefaultQueue<TDev>;
        Queue queue(dev);
        //-----------------------------------------------------------------------------
        alpaka::test::mem::view::testViewMutable<
            TAcc>(
                queue,
                view);
    }

    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TElem>
    auto testViewPlainPtr()
    -> void
    {
        using Dev = alpaka::dev::Dev<TAcc>;
        using Pltf = alpaka::pltf::Pltf<Dev>;

        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;
        using View = alpaka::mem::view::ViewPlainPtr<Dev, TElem, Dim, Idx>;

        Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));

        auto const extentBuf(alpaka::vec::createVecFromIndexedFnWorkaround<Dim, Idx, alpaka::test::CreateExtentBufVal>(Idx()));
        auto buf(alpaka::mem::buf::alloc<TElem, Idx>(dev, extentBuf));

        auto const extentView(extentBuf);
        auto const offsetView(alpaka::vec::Vec<Dim, Idx>::all(static_cast<Idx>(0)));
        View view(
            alpaka::mem::view::getPtrNative(buf),
            alpaka::dev::getDev(buf),
            alpaka::extent::getExtentVec(buf),
            alpaka::mem::view::getPitchBytesVec(buf));

        alpaka::test::mem::view::testViewPlainPtrMutable<TAcc>(view, dev, extentView, offsetView);
    }

    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TElem>
    auto testViewPlainPtrConst()
    -> void
    {
        using Dev = alpaka::dev::Dev<TAcc>;
        using Pltf = alpaka::pltf::Pltf<Dev>;

        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;
        using View = alpaka::mem::view::ViewPlainPtr<Dev, TElem, Dim, Idx>;

        Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));

        auto const extentBuf(alpaka::vec::createVecFromIndexedFnWorkaround<Dim, Idx, alpaka::test::CreateExtentBufVal>(Idx()));
        auto buf(alpaka::mem::buf::alloc<TElem, Idx>(dev, extentBuf));

        auto const extentView(extentBuf);
        auto const offsetView(alpaka::vec::Vec<Dim, Idx>::all(static_cast<Idx>(0)));
        View const view(
            alpaka::mem::view::getPtrNative(buf),
            alpaka::dev::getDev(buf),
            alpaka::extent::getExtentVec(buf),
            alpaka::mem::view::getPitchBytesVec(buf));

        alpaka::test::mem::view::testViewPlainPtrImmutable<TAcc>(view, dev, extentView, offsetView);
    }

    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TElem>
    auto testViewPlainPtrOperators()
    -> void
    {
        using Dev = alpaka::dev::Dev<TAcc>;
        using Pltf = alpaka::pltf::Pltf<Dev>;

        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;
        using View = alpaka::mem::view::ViewPlainPtr<Dev, TElem, Dim, Idx>;

        Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));

        auto const extentBuf(alpaka::vec::createVecFromIndexedFnWorkaround<Dim, Idx, alpaka::test::CreateExtentBufVal>(Idx()));
        auto buf(alpaka::mem::buf::alloc<TElem, Idx>(dev, extentBuf));

        View view(
            alpaka::mem::view::getPtrNative(buf),
            alpaka::dev::getDev(buf),
            alpaka::extent::getExtentVec(buf),
            alpaka::mem::view::getPitchBytesVec(buf));

        // copy-constructor
        View viewCopy(view);

        // move-constructor
        View viewMove(std::move(viewCopy));
    }
}
}
}
}
#if BOOST_COMP_GNUC
    #pragma GCC diagnostic pop
#endif

BOOST_AUTO_TEST_SUITE(memView)

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    viewPlainPtrTest,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    alpaka::test::mem::view::testViewPlainPtr<TAcc, float>();
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    viewPlainPtrConstTest,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    alpaka::test::mem::view::testViewPlainPtrConst<TAcc, float>();
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    viewPlainPtrOperatorTest,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    alpaka::test::mem::view::testViewPlainPtrOperators<TAcc, float>();
}

BOOST_AUTO_TEST_SUITE_END()
