/**
 * \file
 * Copyright 2015-2018 Benjamin Worpitz
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
        typename TIdx,
        typename TBuf>
    auto testViewSubViewImmutable(
        alpaka::mem::view::ViewSubView<TDev, TElem, TDim, TIdx> const & view,
        TBuf & buf,
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

        //-----------------------------------------------------------------------------
        // alpaka::mem::view::traits::GetPitchBytes
        // The pitch of the view has to be identical to the pitch of the underlying buffer in all dimensions.
        {
            auto const pitchBuf(alpaka::mem::view::getPitchBytesVec(buf));
            auto const pitchView(alpaka::mem::view::getPitchBytesVec(view));

            for(TIdx i = TDim::value; i > static_cast<TIdx>(0u); --i)
            {
                BOOST_REQUIRE_EQUAL(
                    pitchBuf[i-static_cast<TIdx>(1u)],
                    pitchView[i-static_cast<TIdx>(1u)]);
            }
        }

        //-----------------------------------------------------------------------------
        // alpaka::mem::view::traits::GetPtrNative
        // The native pointer has to be exactly the value we calculate here.
        {
            auto viewPtrNative(
                reinterpret_cast<std::uint8_t *>(
                    alpaka::mem::view::getPtrNative(buf)));
            auto const pitchBuf(alpaka::mem::view::getPitchBytesVec(buf));
            for(TIdx i = TDim::value; i > static_cast<TIdx>(0u); --i)
            {
                auto const pitch = (i < static_cast<TIdx>(TDim::value)) ? pitchBuf[i] : static_cast<TIdx>(sizeof(TElem));
                viewPtrNative += offsetView[i - static_cast<TIdx>(1u)] * pitch;
            }
            BOOST_REQUIRE_EQUAL(
                reinterpret_cast<TElem *>(viewPtrNative),
                alpaka::mem::view::getPtrNative(view));
        }
    }

    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TDev,
        typename TElem,
        typename TDim,
        typename TIdx,
        typename TBuf>
    auto testViewSubViewMutable(
        alpaka::mem::view::ViewSubView<TDev, TElem, TDim, TIdx> & view,
        TBuf & buf,
        TDev const & dev,
        alpaka::vec::Vec<TDim, TIdx> const & extentView,
        alpaka::vec::Vec<TDim, TIdx> const & offsetView)
    -> void
    {
        //-----------------------------------------------------------------------------
        testViewSubViewImmutable<
            TAcc>(
                view,
                buf,
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
    auto testViewSubViewNoOffset()
    -> void
    {
        using Dev = alpaka::dev::Dev<TAcc>;
        using Pltf = alpaka::pltf::Pltf<Dev>;

        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;
        using View = alpaka::mem::view::ViewSubView<Dev, TElem, Dim, Idx>;

        Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));

        auto const extentBuf(alpaka::vec::createVecFromIndexedFnWorkaround<Dim, Idx, alpaka::test::CreateExtentBufVal>(Idx()));
        auto buf(alpaka::mem::buf::alloc<TElem, Idx>(dev, extentBuf));

        auto const extentView(extentBuf);
        auto const offsetView(alpaka::vec::Vec<Dim, Idx>::all(static_cast<Idx>(0)));
        View view(buf);

        alpaka::test::mem::view::testViewSubViewMutable<TAcc>(view, buf, dev, extentView, offsetView);
    }

    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TElem>
    auto testViewSubViewOffset()
    -> void
    {
        using Dev = alpaka::dev::Dev<TAcc>;
        using Pltf = alpaka::pltf::Pltf<Dev>;

        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;
        using View = alpaka::mem::view::ViewSubView<Dev, TElem, Dim, Idx>;

        Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));

        auto const extentBuf(alpaka::vec::createVecFromIndexedFnWorkaround<Dim, Idx, alpaka::test::CreateExtentBufVal>(Idx()));
        auto buf(alpaka::mem::buf::alloc<TElem, Idx>(dev, extentBuf));

        auto const extentView(alpaka::vec::createVecFromIndexedFnWorkaround<Dim, Idx, alpaka::test::CreateExtentViewVal>(Idx()));
        auto const offsetView(alpaka::vec::Vec<Dim, Idx>::all(static_cast<Idx>(1)));
        View view(buf, extentView, offsetView);

        alpaka::test::mem::view::testViewSubViewMutable<TAcc>(view, buf, dev, extentView, offsetView);
    }

    //-----------------------------------------------------------------------------
    template<
        typename TAcc,
        typename TElem>
    auto testViewSubViewOffsetConst()
    -> void
    {
        using Dev = alpaka::dev::Dev<TAcc>;
        using Pltf = alpaka::pltf::Pltf<Dev>;

        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;
        using View = alpaka::mem::view::ViewSubView<Dev, TElem, Dim, Idx>;

        Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));

        auto const extentBuf(alpaka::vec::createVecFromIndexedFnWorkaround<Dim, Idx, CreateExtentBufVal>(Idx()));
        auto buf(alpaka::mem::buf::alloc<TElem, Idx>(dev, extentBuf));

        auto const extentView(alpaka::vec::createVecFromIndexedFnWorkaround<Dim, Idx, CreateExtentViewVal>(Idx()));
        auto const offsetView(alpaka::vec::Vec<Dim, Idx>::all(static_cast<Idx>(1)));
        View const view(buf, extentView, offsetView);

        alpaka::test::mem::view::testViewSubViewImmutable<TAcc>(view, buf, dev, extentView, offsetView);
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
    viewSubViewNoOffsetTest,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    alpaka::test::mem::view::testViewSubViewNoOffset<TAcc, float>();
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    viewSubViewOffsetTest,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    alpaka::test::mem::view::testViewSubViewOffset<TAcc, float>();
}

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    viewSubViewOffsetConstTest,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    alpaka::test::mem::view::testViewSubViewOffsetConst<TAcc, float>();
}

BOOST_AUTO_TEST_SUITE_END()
