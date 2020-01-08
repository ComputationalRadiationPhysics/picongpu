/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/idx/Accessors.hpp>
#include <alpaka/idx/MapIdx.hpp>

#include <alpaka/meta/ForEachType.hpp>
#include <alpaka/test/dim/TestDims.hpp>

#include <catch2/catch.hpp>

//#############################################################################
//! 1D: (17)
//! 2D: (17, 14)
//! 3D: (17, 14, 11)
//! 4D: (17, 14, 11, 8)
template<
    std::size_t Tidx>
struct CreateExtentVal
{
    //-----------------------------------------------------------------------------
    template<
        typename TIdx>
    ALPAKA_FN_HOST_ACC static auto create(
        TIdx)
    -> TIdx
    {
        return  static_cast<TIdx>(17u - (Tidx*3u));
    }
};

//-----------------------------------------------------------------------------
struct TestTemplate
{
template< typename TDim >
void operator()()
{
    using Idx = std::size_t;
    using Vec = alpaka::vec::Vec<TDim, Idx>;

    auto const extentNd(alpaka::vec::createVecFromIndexedFnWorkaround<TDim, Idx, CreateExtentVal>(Idx()));
    auto const idxNd(extentNd - Vec::all(4u));

    auto const idx1d(alpaka::idx::mapIdx<1u>(idxNd, extentNd));

    auto const idxNdResult(alpaka::idx::mapIdx<TDim::value>(idx1d, extentNd));

    REQUIRE(idxNd == idxNdResult);
}
};

TEST_CASE( "mapIdx", "[idx]")
{
    alpaka::meta::forEachType< alpaka::test::dim::TestDims >( TestTemplate() );
}
