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

#include "picongpu/MetadataAggregator.hpp"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

using nlohmann::json;
using picongpu::addMetadataOf;
using picongpu::MetadataAggregator;

// doc-include-start: adapting metadata
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
// doc-include-end: adapting metadata

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

    json description() const
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

// doc-include-start: reusing default metadata
template<>
struct picongpu::traits::GetMetadata<SomethingWithPrivateInfo>
{
    SomethingWithPrivateInfo const& obj;

    json description() const
    {
        auto result = obj.metadata();
        // could also depend on the (publicly accessible members of) `obj`:
        result["customisedInfo"] = "Some customised string.";
        return result;
    }
};
// doc-include-end: reusing default metadata

// doc-include-start: declare metadata as friend
struct SomethingWithoutUsefulMetadata : SomethingWithPrivateInfo
{
    friend picongpu::traits::GetMetadata<SomethingWithoutUsefulMetadata>;
    // ...
    // doc-include-end: declare metadata as friend

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

    json description() const
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

struct SomethingWithCustomCTInfo : SomethingWithCTInfo
{
};

template<>
struct picongpu::traits::GetMetadata<SomethingWithCustomCTInfo>
{
    static json description()
    {
        // Maybe the value of info needs to be multiplied by a unit to get a meaningful value for metadata:
        int unit = 10;
        json result = SomethingWithCustomCTInfo::metadata();
        // It's surprisingly verbose to get back an int from this. See https://json.nlohmann.me for more details.
        result["Info"] = result.at("Info").get<int>() * unit;
        return result;
    }
};

// An example for the documentation:

struct CompileTimeInformation
{
    static constexpr int value = 42;
};

struct MyClass
{
    using MyCompileTimeInformation = CompileTimeInformation;
    const int runtimeValue = 8;
    // would normally also provide a default implementation of
    // json description() const;
};

// doc-include-start: metadata customisation
template<>
struct picongpu::traits::GetMetadata<MyClass>
{
    MyClass const& obj;

    json description() const
    {
        json result = json::object(); // always use objects and not arrays as root
        result["my"]["cool"]["runtimeValue"] = obj.runtimeValue;
        result["my"]["cool"]["compiletimeValue"] = MyClass::MyCompileTimeInformation::value;
        result["somethingElseThatSeemedImportant"] = "not necessarily derived from obj or MyClass";
        return result;
    }
};
// doc-include-end: metadata customisation

TEST_CASE("unit::metadataDescription", "[metadata description test]")
{
    MetadataAggregator metadataAggregator;

    SECTION("can add RT info for simple object")
    {
        SomethingWithRTInfo obj{42};
        auto expected = json::object();
        expected["info"] = obj.info;

        addMetadataOf(obj);
        CHECK(metadataAggregator.metadata == expected);
    }

    SECTION("can handle multiple pieces of information")
    {
        SomethingWithMoreRTInfo obj{42, 'j'};
        auto expected = json::object();
        expected["moreInfo"] = obj.moreInfo;
        expected["character"] = obj.c;

        addMetadataOf(obj);
        CHECK(metadataAggregator.metadata == expected);
    }

    SECTION("metadata can be customised via trait")
    {
        SomethingWithCustomRTInfo obj{42};
        auto expected = json::object();
        expected["info"] = obj.moreInfo;
        REQUIRE(obj.metadata() != expected); // make sure we test something non-trivial here

        addMetadataOf(obj);
        CHECK(metadataAggregator.metadata == expected);
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
        CHECK(metadataAggregator.metadata == expected);
    }

    SECTION("overlapping metadata overwrites (test for confidence and documentation)")
    {
        // actually in the documentation we explicitly don't specify an order, so this is technically undefined
        // behaviour
        SomethingWithRTInfo obj{42};
        SomethingWithCustomRTInfo obj2{41, 'j'}; // provides same entry as previous one
        auto expected = json::object();
        expected["info"] = obj2.moreInfo;

        addMetadataOf(obj);
        addMetadataOf(obj2);
        CHECK(metadataAggregator.metadata == expected);
    }

    SECTION("customisation can extract private information from `.metadata()`")
    {
        int privateInfo = 42;
        SomethingWithPrivateInfo obj{privateInfo};
        auto expected = json::object();
        expected["privateInfo"] = privateInfo;
        expected["customisedInfo"] = "Some customised string.";
        REQUIRE(obj.metadata() != expected); // make sure we test something non-trivial here

        addMetadataOf(obj);
        CHECK(metadataAggregator.metadata == expected);
    }

    SECTION("customisation can extract private information as a friend")
    {
        int privateInfo = 42;
        SomethingWithoutUsefulMetadata obj{privateInfo};
        auto expected = json::object();
        expected["privateInfo"] = privateInfo;
        expected["customisedInfo"] = "Some other customised string.";
        // make sure we test something non-trivial here
        REQUIRE(obj.metadata() != expected);

        addMetadataOf(obj);
        CHECK(metadataAggregator.metadata == expected);
    }

    SECTION("can extract default CT information")
    {
        auto expected = json::object();
        expected["Info"] = SomethingWithCTInfo::Info::info;

        addMetadataOf<SomethingWithCTInfo>();
        CHECK(metadataAggregator.metadata == expected);
    }

    SECTION("can extract customised CT information")
    {
        auto expected = json::object();
        expected["Info"] = 10 * SomethingWithCustomCTInfo::Info::info;

        addMetadataOf<SomethingWithCTInfo>();
        CHECK(metadataAggregator.metadata == expected);
    }

    SECTION("check documentation example")
    {
        MyClass obj;
        auto expected = json::object();
        expected["my"]["cool"]["runtimeValue"] = obj.runtimeValue;
        expected["my"]["cool"]["compiletimeValue"] = MyClass::MyCompileTimeInformation::value;
        expected["somethingElseThatSeemedImportant"] = "not necessarily derived from obj or MyClass";

        addMetadataOf(obj);
        CHECK(metadataAggregator.metadata == expected);
    }

    MetadataAggregator::metadata = json::object();
}
