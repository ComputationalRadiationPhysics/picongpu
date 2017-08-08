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

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>  // alpaka::test::acc::TestAccs

#include <boost/predef.h>           // BOOST_COMP_CLANG
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

BOOST_AUTO_TEST_SUITE(idx)

//#############################################################################
//! 1D: (17)
//! 2D: (17, 14)
//! 3D: (17, 14, 11)
//! 4D: (17, 14, 11, 8)
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
        return  static_cast<TSize>(17u - (Tidx*3u));
    }
};

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    mapIdx,
    TDim,
    alpaka::test::acc::TestDims)
{
    using Size = std::size_t;
    using Vec = alpaka::vec::Vec<TDim, Size>;

    auto const extentNd(alpaka::vec::createVecFromIndexedFnWorkaround<TDim, Size, CreateExtentBufVal>(Size()));
    auto const idxNd(extentNd - Vec::all(4u));

    auto const idx1d(alpaka::idx::mapIdx<1u>(idxNd, extentNd));

    auto const idxNdResult(alpaka::idx::mapIdx<TDim::value>(idx1d, extentNd));

    BOOST_REQUIRE_EQUAL(idxNd, idxNdResult);
}

BOOST_AUTO_TEST_SUITE_END()
