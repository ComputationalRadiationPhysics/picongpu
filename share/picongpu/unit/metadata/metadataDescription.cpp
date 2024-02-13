/* Copyright 2023 Julian Lenz
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <pmacc/boost_workaround.hpp>

#include "picongpu/metadata.hpp"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

using nlohmann::json;
using picongpu::addMetadataOf;
using picongpu::MetadataPlugin;

struct SomethingWithRTInfo
{
    int info = 0;
};

TEST_CASE("unit::metadataDescription", "[metadata description test]")
{
    MetadataPlugin metadataPlugin;

    SECTION("can add RT info for simple object")
    {
        SomethingWithRTInfo obj{42};
        json expected = json::object({{"info", 42}});

        addMetadataOf(obj);
        CHECK(metadataPlugin.metadata == expected);
    }
}
