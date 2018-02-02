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

BOOST_AUTO_TEST_SUITE(memView)

//#############################################################################
//! 1D: sizeof(TSize) * (5)
//! 2D: sizeof(TSize) * (5, 4)
//! 3D: sizeof(TSize) * (5, 4, 3)
//! 4D: sizeof(TSize) * (5, 4, 3, 2)
//#############################################################################
template<
    std::size_t Tidx>
struct CreateExtentBufVal
{
    //-----------------------------------------------------------------------------
    //!
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

//#############################################################################
//! 1D: sizeof(TSize) * (4)
//! 2D: sizeof(TSize) * (4, 3)
//! 3D: sizeof(TSize) * (4, 3, 2)
//! 4D: sizeof(TSize) * (4, 3, 2, 1)
template<
    std::size_t Tidx>
struct CreateExtentViewVal
{
    //-----------------------------------------------------------------------------
    template<
        typename TSize>
    static auto create(
        TSize)
    -> TSize
    {
        return sizeof(TSize) * (4u - Tidx);
    }
};

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    viewSubViewTest,
    TAcc,
    alpaka::test::acc::TestAccs)
{
    using Dev = alpaka::dev::Dev<TAcc>;
    using Pltf = alpaka::pltf::Pltf<Dev>;
    using Stream = alpaka::test::stream::DefaultStream<Dev>;

    using Elem = float;
    using Dim = alpaka::dim::Dim<TAcc>;
    using Size = alpaka::size::Size<TAcc>;
    using View = alpaka::mem::view::ViewSubView<Dev, Elem, Dim, Size>;

    Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));
    Stream stream(dev);

    // We have to be careful with the extents used.
    // When Size is a 8 bit signed integer and Dim is 4, the extent is extremely limited.
    auto const extentBuf(alpaka::vec::createVecFromIndexedFnWorkaround<Dim, Size, CreateExtentBufVal>(Size()));
    auto buf(alpaka::mem::buf::alloc<Elem, Size>(dev, extentBuf));

    // TODO: Test failing cases of view extents larger then the underlying buffer extents.
    auto const extentView(alpaka::vec::createVecFromIndexedFnWorkaround<Dim, Size, CreateExtentViewVal>(Size()));
    auto const offsetView(alpaka::vec::Vec<Dim, Size>::all(sizeof(Size)));
    View view(buf, extentView, offsetView);

    //-----------------------------------------------------------------------------
    alpaka::test::mem::view::viewTestImmutable<
        Elem>(
            view,
            dev,
            extentView,
            offsetView);

    //-----------------------------------------------------------------------------
    alpaka::test::mem::view::viewTestMutable<
        TAcc>(
            stream,
            view);

    //-----------------------------------------------------------------------------
    // alpaka::mem::view::traits::GetPitchBytes
    // The pitch of the view has to be identical to the pitch of the underlying buffer in all dimensions.
    {
        auto const pitchBuf(alpaka::mem::view::getPitchBytesVec(buf));
        auto const pitchView(alpaka::mem::view::getPitchBytesVec(view));

        for(Size i = Dim::value; i > static_cast<Size>(0u); --i)
        {
            BOOST_REQUIRE_EQUAL(
                pitchBuf[i-static_cast<Size>(1u)],
                pitchView[i-static_cast<Size>(1u)]);
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
        for(Size i = Dim::value; i > static_cast<Size>(0u); --i)
        {
            auto const pitch = (i < static_cast<Size>(Dim::value)) ? pitchBuf[i] : static_cast<Size>(sizeof(Elem));
            viewPtrNative += offsetView[i - static_cast<Size>(1u)] * pitch;
        }
        BOOST_REQUIRE_EQUAL(
            reinterpret_cast<Elem *>(viewPtrNative),
            alpaka::mem::view::getPtrNative(view));
    }
}

BOOST_AUTO_TEST_SUITE_END()
