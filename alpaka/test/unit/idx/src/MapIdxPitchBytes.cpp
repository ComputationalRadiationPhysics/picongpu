/* Copyright 2022 Jeffrey Kelling, Bernhard Manfred Gruber, Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/dev/Traits.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <alpaka/idx/Accessors.hpp>
#include <alpaka/idx/MapIdx.hpp>
#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/mem/view/ViewSubView.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/dim/TestDims.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

TEMPLATE_LIST_TEST_CASE("mapIdxPitchBytes", "[idx]", alpaka::test::TestDims)
{
    using Dim = TestType;
    using Idx = std::size_t;
    using Vec = alpaka::Vec<Dim, Idx>;

    auto const extentNd
        = alpaka::createVecFromIndexedFn<Dim, alpaka::test::CreateVecWithIdx<Idx>::template ForExtentBuf>();

    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    using Elem = std::uint8_t;
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto parentView = alpaka::createView(devAcc, static_cast<Elem*>(nullptr), extentNd);

    auto const offset = Vec::all(4u);
    auto const extent = Vec::all(4u);
    auto const idxNd = Vec::all(2u);
    auto view = alpaka::createSubView(parentView, extent, offset);
    auto pitch = alpaka::getPitchBytesVec(view);

    auto const idx1d = alpaka::mapIdxPitchBytes<1u>(idxNd, pitch);
    auto const idx1dDelta = alpaka::mapIdx<1u>(idxNd + offset, extentNd) - alpaka::mapIdx<1u>(offset, extentNd);

    auto const idxNdResult = alpaka::mapIdxPitchBytes<Dim::value>(idx1d, pitch);

    // linear index in pitched offset box should be the difference between
    // linear index in parent box and linear index of offset
    REQUIRE(idx1d == idx1dDelta);
    // roundtrip
    REQUIRE(idxNd == idxNdResult);
}
