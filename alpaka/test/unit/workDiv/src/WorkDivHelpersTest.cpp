/* Copyright 2020 Sergei Bastrakov
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/workdiv/WorkDivHelpers.hpp>

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>

#include <catch2/catch.hpp>

 //-----------------------------------------------------------------------------
namespace
{
    template< typename TAcc >
    auto getWorkDiv()
    {
        using Dev = alpaka::dev::Dev<TAcc>;
        using Pltf = alpaka::pltf::Pltf<Dev>;
        using Dim = alpaka::dim::Dim<TAcc>;
        using Idx = alpaka::idx::Idx<TAcc>;

        Dev const dev(alpaka::pltf::getDevByIdx<Pltf>(0u));
        auto const gridThreadExtent = alpaka::vec::Vec<Dim, Idx>::all(10);
        auto const threadElementExtent = alpaka::vec::Vec<Dim, Idx>::ones();
        auto workDiv = alpaka::workdiv::getValidWorkDiv<TAcc>(
            dev,
            gridThreadExtent,
            threadElementExtent,
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted);
        return workDiv;
    }
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "getValidWorkDiv", "[workDiv]", alpaka::test::acc::TestAccs)
{
    using Acc = TestType;
    // Note: getValidWorkDiv() is called inside getWorkDiv
    auto workDiv = getWorkDiv< Acc >();
    alpaka::ignore_unused( workDiv );
}

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "isValidWorkDiv", "[workDiv]", alpaka::test::acc::TestAccs)
{
    using Acc = TestType;
    using Dev = alpaka::dev::Dev<Acc>;
    using Pltf = alpaka::pltf::Pltf<Dev>;

    Dev dev(alpaka::pltf::getDevByIdx<Pltf>(0u));
    auto workDiv = getWorkDiv< Acc >();
    // Test both overloads
    REQUIRE( alpaka::workdiv::isValidWorkDiv(
        alpaka::acc::getAccDevProps< Acc >( dev ),
        workDiv));
    REQUIRE( alpaka::workdiv::isValidWorkDiv<Acc>(
        dev,
        workDiv));
}
