/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/idx/Accessors.hpp>
#include <alpaka/idx/MapIdx.hpp>
#include <alpaka/meta/ForEachType.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/dim/TestDims.hpp>

#include <catch2/catch.hpp>

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE("mapIdx", "[idx]", alpaka::test::TestDims)
{
    using Dim = TestType;
    using Idx = std::size_t;
    using Vec = alpaka::Vec<Dim, Idx>;

    auto const extentNd(
        alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>());
    auto const idxNd(extentNd - Vec::all(4u));

    auto const idx1d(alpaka::mapIdx<1u>(idxNd, extentNd));

    auto const idxNdResult(alpaka::mapIdx<Dim::value>(idx1d, extentNd));

    REQUIRE(idxNd == idxNdResult);
}
