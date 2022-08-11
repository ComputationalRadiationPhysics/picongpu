/* Copyright 2022 Sergei Bastrakov, Bernhard Manfred Gruber, Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/acc/AccDevProps.hpp>
#include <alpaka/acc/Traits.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

TEMPLATE_LIST_TEST_CASE("getAccDevProps", "[acc]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    using Dev = alpaka::Dev<Acc>;
    using Pltf = alpaka::Pltf<Dev>;
    Dev const dev(alpaka::getDevByIdx<Pltf>(0u));
    auto const devProps = alpaka::getAccDevProps<Acc>(dev);

    REQUIRE(devProps.m_gridBlockExtentMax.min() > 0);
    REQUIRE(devProps.m_blockThreadExtentMax.min() > 0);
    REQUIRE(devProps.m_threadElemExtentMax.min() > 0);
    REQUIRE(devProps.m_gridBlockCountMax > 0);
    REQUIRE(devProps.m_blockThreadCountMax > 0);
    REQUIRE(devProps.m_threadElemCountMax > 0);
    REQUIRE(devProps.m_multiProcessorCount > 0);
    REQUIRE(devProps.m_sharedMemSizeBytes > 0);
}
