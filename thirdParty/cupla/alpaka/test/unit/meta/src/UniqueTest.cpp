/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/meta/Unique.hpp>

#include <catch2/catch_test_macros.hpp>

#include <tuple>
#include <type_traits>

TEST_CASE("uniqueWithDuplicate", "[meta]")
{
    using UniqueInput = std::tuple<int, float, int, float, float, int>;

    using UniqueResult = alpaka::meta::Unique<UniqueInput>;

    using UniqueReference = std::tuple<int, float>;

    static_assert(std::is_same_v<UniqueReference, UniqueResult>, "alpaka::meta::Unique failed!");
}

TEST_CASE("uniqueWithoutDuplicate", "[meta]")
{
    using UniqueInput = std::tuple<int, float, double>;

    using UniqueResult = alpaka::meta::Unique<UniqueInput>;

    using UniqueReference = UniqueInput;

    static_assert(std::is_same_v<UniqueReference, UniqueResult>, "alpaka::meta::Unique failed!");
}
