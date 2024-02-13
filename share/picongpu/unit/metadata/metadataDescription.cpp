/* Copyright 2024 Julian Lenz
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

json picongpu::MetadataPlugin::metadata;

struct SomethingWithRTInfo
{
    int info = 0;

    json metadata() const
    {
        auto result = json::object();
        result["info"] = info;
        return result;
    }
};

struct SomethingWithMoreRTInfo
{
    int info = 0;
    char c = 'a';

    json metadata() const
    {
        auto result = json::object();
        result["info"] = info;
        result["character"] = c;
        return result;
    }
};

struct SomethingWithCustomRTInfo : SomethingWithMoreRTInfo
{
    // We simply derive from a parent because we need a `.metadata()` implementation to override but we're not really
    // interested in what it is.
};

template<>
struct picongpu::traits::GetMetadata<SomethingWithCustomRTInfo>
{
    SomethingWithCustomRTInfo const& obj;

    json description() const
    {
        auto result = json::object();
        result["info"] = obj.info;
        // Is different from output of .metadata() because we are not reporting `c`.
        return result;
    }
};


TEST_CASE("unit::metadataDescription", "[metadata description test]")
{
    MetadataPlugin metadataPlugin;

    SECTION("can add RT info for simple object")
    {
        SomethingWithRTInfo obj{42};
        auto expected = json::object();
        expected["info"] = obj.info;

        addMetadataOf(obj);
        CHECK(metadataPlugin.metadata == expected);
    }

    SECTION("can add RT info for simple object")
    {
        SomethingWithMoreRTInfo obj{42, 'j'};
        auto expected = json::object();
        expected["info"] = obj.info;
        expected["character"] = obj.c;

        addMetadataOf(obj);
        CHECK(metadataPlugin.metadata == expected);
    }

    SECTION("metadata can be customised via trait")
    {
        SomethingWithCustomRTInfo obj{42};
        auto expected = json::object();
        expected["info"] = obj.info;
        REQUIRE(obj.metadata() != expected); // make sure we test something non-trivial here

        addMetadataOf(obj);
        CHECK(metadataPlugin.metadata == expected);
    }
}
