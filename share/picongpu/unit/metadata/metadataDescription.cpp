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
    int moreInfo = 0;
    char c = 'a';

    json metadata() const
    {
        auto result = json::object();
        result["moreInfo"] = moreInfo;
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

    json descriptionRT() const
    {
        auto result = json::object();
        result["info"] = obj.moreInfo;
        // Is different from output of .metadata() because we are not reporting `c`.
        return result;
    }
};

struct SomethingWithPrivateInfo
{
protected:
    int privateInfo = 0;

public:
    SomethingWithPrivateInfo(int i) : privateInfo{i}
    {
    }

    virtual json metadata() const
    {
        auto result = json::object();
        result["privateInfo"] = privateInfo;
        return result;
    }
};

template<>
struct picongpu::traits::GetMetadata<SomethingWithPrivateInfo>
{
    SomethingWithPrivateInfo const& obj;

    json descriptionRT() const
    {
        auto result = obj.metadata();
        result["customisedInfo"] = "Some customised string.";
        return result;
    }
};

struct SomethingWithoutUsefulMetadata : SomethingWithPrivateInfo
{
    friend picongpu::traits::GetMetadata<SomethingWithoutUsefulMetadata>;

    SomethingWithoutUsefulMetadata(int i) : SomethingWithPrivateInfo(i){};
    json metadata() const override
    {
        return {};
    }
};

template<>
struct picongpu::traits::GetMetadata<SomethingWithoutUsefulMetadata>
{
    SomethingWithoutUsefulMetadata const& obj;

    json descriptionRT() const
    {
        json result = json::object();
        result["privateInfo"] = obj.privateInfo;
        result["customisedInfo"] = "Some other customised string.";
        return result;
    }
};

struct SomeParameter
{
    static constexpr int info = 0;
};

struct SomethingWithCTInfo
{
    using Info = SomeParameter;
    static json metadata()
    {
        json result = json::object();
        result["Info"] = Info::info;
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
        expected["moreInfo"] = obj.moreInfo;
        expected["character"] = obj.c;

        addMetadataOf(obj);
        CHECK(metadataPlugin.metadata == expected);
    }

    SECTION("metadata can be customised via trait")
    {
        SomethingWithCustomRTInfo obj{42};
        auto expected = json::object();
        expected["info"] = obj.moreInfo;
        REQUIRE(obj.metadata() != expected); // make sure we test something non-trivial here

        addMetadataOf(obj);
        CHECK(metadataPlugin.metadata == expected);
    }

    SECTION("merges two non-overlapping sets of metadata")
    {
        SomethingWithRTInfo obj{42};
        SomethingWithMoreRTInfo obj2{42, 'j'};
        auto expected = json::object();
        expected["info"] = obj.info;
        expected["moreInfo"] = obj2.moreInfo;
        expected["character"] = obj2.c;

        addMetadataOf(obj);
        addMetadataOf(obj2);
        CHECK(metadataPlugin.metadata == expected);
    }

    SECTION("overlapping metadata overwrites (test for confidence and documentation)")
    {
        SomethingWithRTInfo obj{42};
        SomethingWithCustomRTInfo obj2{41, 'j'}; // provides same entry as previous one
        auto expected = json::object();
        expected["info"] = obj2.moreInfo;

        addMetadataOf(obj);
        addMetadataOf(obj2);
        CHECK(metadataPlugin.metadata == expected);
    }

    SECTION("customisation can extract private information from `.metadata()`")
    {
        int privateInfo = 42;
        SomethingWithPrivateInfo obj{privateInfo};
        auto expected = json::object();
        expected["privateInfo"] = privateInfo;
        expected["customisedInfo"] = "Some customised string.";

        addMetadataOf(obj);
        CHECK(metadataPlugin.metadata == expected);
    }

    SECTION("customisation can extract private information as a friend")
    {
        int privateInfo = 42;
        SomethingWithoutUsefulMetadata obj{privateInfo};
        auto expected = json::object();
        expected["privateInfo"] = privateInfo;
        expected["customisedInfo"] = "Some other customised string.";

        addMetadataOf(obj);
        CHECK(metadataPlugin.metadata == expected);
    }

    SECTION("can extract default CT information")
    {
        auto expected = json::object();
        expected["Info"] = SomethingWithCTInfo::Info::info;

        addMetadataOf<SomethingWithCTInfo>();
        CHECK(metadataPlugin.metadata == expected);
    }
    MetadataPlugin::metadata = json::object();
}
