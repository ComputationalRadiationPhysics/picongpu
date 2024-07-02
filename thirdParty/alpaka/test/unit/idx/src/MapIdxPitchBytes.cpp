/* Copyright 2022 Jeffrey Kelling, Bernhard Manfred Gruber, Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/dev/Traits.hpp>
#include <alpaka/idx/Accessors.hpp>
#include <alpaka/idx/MapIdx.hpp>
#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/mem/view/ViewSubView.hpp>
#include <alpaka/test/Extent.hpp>
#include <alpaka/test/dim/TestDims.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

template<typename TDim, typename TAccTag>
auto mapIdxPitchBytes(TAccTag const&)
{
    using Dim = TDim;
    using Idx = std::size_t;
    using Vec = alpaka::Vec<Dim, Idx>;

    auto const extentNd = alpaka::test::extentBuf<Dim, Idx>;

    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    using Elem = std::uint8_t;
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);
    auto parentView = alpaka::createView(devAcc, static_cast<Elem*>(nullptr), extentNd);

    auto const offset = Vec::all(4u);
    auto const extent = Vec::all(4u);
    auto const idxNd = Vec::all(2u);
    auto view = alpaka::createSubView(parentView, extent, offset);
    auto pitch = alpaka::getPitchesInBytes(view);

    auto const idx1d = alpaka::mapIdxPitchBytes<1u>(idxNd, pitch);
    auto const idx1dDelta = alpaka::mapIdx<1u>(idxNd + offset, extentNd) - alpaka::mapIdx<1u>(offset, extentNd);

    auto const idxNdResult = alpaka::mapIdxPitchBytes<Dim::value>(idx1d, pitch);

    // linear index in pitched offset box should be the difference between
    // linear index in parent box and linear index of offset
    REQUIRE(idx1d == idx1dDelta);
    // roundtrip
    REQUIRE(idxNd == idxNdResult);
}

TEMPLATE_LIST_TEST_CASE("mapIdxPitchBytes", "[idx]", alpaka::test::NonZeroTestDims)
{
    // execute the example once for each enabled accelerator
    std::apply([](auto const&... tags) { (mapIdxPitchBytes<TestType>(tags), ...); }, alpaka::EnabledAccTags{});
}
