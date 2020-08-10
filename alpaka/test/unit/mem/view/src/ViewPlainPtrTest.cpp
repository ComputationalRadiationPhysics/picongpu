/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Erik Zenker
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/mem/view/ViewPlainPtr.hpp>

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/test/mem/view/ViewTest.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/meta/ForEachType.hpp>
#include <alpaka/core/BoostPredef.hpp>

#include <catch2/catch.hpp>

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

        auto const extentBuf(alpaka::vec::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>());
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

        auto const extentBuf(alpaka::vec::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>());
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

        auto const extentBuf(alpaka::vec::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>());
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

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "viewPlainPtrTest", "[memView]", alpaka::test::acc::TestAccs)
{
    alpaka::test::mem::view::testViewPlainPtr<TestType, float>();
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "viewPlainPtrConstTest", "[memView]", alpaka::test::acc::TestAccs)
{
    alpaka::test::mem::view::testViewPlainPtrConst<TestType, float>();
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "viewPlainPtrOperatorTest", "[memView]", alpaka::test::acc::TestAccs)
{
    alpaka::test::mem::view::testViewPlainPtrOperators<TestType, float>();
}
