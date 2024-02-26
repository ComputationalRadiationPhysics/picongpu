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
#include "picongpu/traits/GetMetadata.hpp"
#include "pmacc/meta/conversion/MakeSeq.hpp"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

using nlohmann::json;
using picongpu::addMetadataOf;
using picongpu::MetadataAggregator;
using picongpu::traits::AllowMissingMetadata;

struct NotHavingMetadata
{
    static constexpr int info = 42;
};

struct HavingMetadata
{
    static constexpr int info = 0;

    static json metadata()
    {
        auto result = json::object();
        result["info"] = info;
        return result;
    }
};

struct HavingCustomisedMetadata : HavingMetadata
{
};

template<>
struct picongpu::traits::GetMetadata<HavingCustomisedMetadata>
{
    json description() const
    {
        auto original = HavingCustomisedMetadata ::metadata();
        original["customised"] = "custom string";
        return original;
    }
};

struct FakeXMin
{
};

TEST_CASE("unit::metadataAllowMissing", "[metadata allow missing test]")
{
    MetadataAggregator metadataAggregator;

    SECTION("AllowMissing")
    {
        SECTION("adding metadata does nothing if no metadata is available")
        {
            auto expected = json::object();

            addMetadataOf<AllowMissingMetadata<NotHavingMetadata>>();
            CHECK(metadataAggregator.metadata == expected);
        }

        SECTION("adding metadata still respects .metadata()")
        {
            auto expected = json::object();
            expected["info"] = HavingMetadata::info;

            addMetadataOf<AllowMissingMetadata<HavingMetadata>>();
            CHECK(metadataAggregator.metadata == expected);
        }

        SECTION("adding metadata still customised trait")
        {
            auto expected = json::object();
            expected["info"] = HavingCustomisedMetadata::info;
            expected["customised"] = "custom string";

            addMetadataOf<AllowMissingMetadata<HavingCustomisedMetadata>>();
            CHECK(metadataAggregator.metadata == expected);
        }
    }

    SECTION("IncidentFieldPolicy")
    {
        SECTION("metadata starts with 'incidentField'")
        {
            using Profiles = pmacc::MakeSeq_t<FakeXMin>;
            addMetadataOf<picongpu::traits::IncidentFieldPolicy<Profiles>>();
            CHECK(metadataAggregator.metadata.contains("incidentField"));
        }
    }

    MetadataAggregator::metadata = json::object();
}
