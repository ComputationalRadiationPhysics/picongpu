/*
 * SPDX-FileCopyrightText: Julian Lenz
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "pmacc/pluginSystem/toSlice.hpp"

#include "catch2/matchers/catch_matchers.hpp"
#include "catch2/matchers/catch_matchers_range_equals.hpp"

#include <algorithm>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

using Catch::Matchers::IsEmpty;
using Catch::Matchers::SizeIs;
using pmacc::pluginSystem::Slice;

using pmacc::pluginSystem::toRangeSlice;
using pmacc::pluginSystem::toTimeSlice;

struct SliceEqual
{
    auto operator()(Slice const& lhs, Slice const& rhs) const
    {
        return std::equal(lhs.values.cbegin(), lhs.values.cend(), rhs.values.cbegin());
    }
};

namespace pmacc::pluginSystem
{
    // This allows catch2 to print usable output in case of a failure.
    template<typename T_Stream>
    auto operator<<(T_Stream& out, Slice const& s)
    {
        out << "{" << s.values[0] << ", " << s.values[1] << ", " << s.values[2] << "}";
    }
} // namespace pmacc::pluginSystem

TEST_CASE("Both slices", "")
{
    std::function<std::vector<Slice>(std::string const&)> toSlice = GENERATE(toTimeSlice, toRangeSlice);

    SECTION("accept empty strings.")
    {
        std::string input = "";
        auto result = toSlice(input);
        CHECK_THAT(result, IsEmpty());
    }

    SECTION("accept : and , syntax.")
    {
        std::vector<std::pair<std::string, std::vector<Slice>>> testCases
            = {// empty
               {"", {}},
               // only colons
               {":", {Slice{0, static_cast<uint32_t>(-1), 1}}},
               {"::", {Slice{0, static_cast<uint32_t>(-1), 1}}},
               // single value
               {":2", {Slice{0, 2, 1}}},
               {":2:", {Slice{0, 2, 1}}},
               {"2:", {Slice{2, static_cast<uint32_t>(-1), 1}}},
               {"2::", {Slice{2, static_cast<uint32_t>(-1), 1}}},
               {"::2", {Slice{0, static_cast<uint32_t>(-1), 2}}},
               // two values
               {"1:5", {Slice{1, 5, 1}}},
               {"1:5:", {Slice{1, 5, 1}}},
               {":1:5", {Slice{0, 1, 5}}},
               {"1::5", {Slice{1, static_cast<uint32_t>(-1), 5}}},
               // three values
               {"1:5:2", {Slice{1, 5, 2}}},
               // explicit -1 as end
               {"1:-1", {Slice{1, static_cast<uint32_t>(-1), 1}}},
               // multiple slices
               {"1:5,10:15", {Slice{1, 5, 1}, Slice{10, 15, 1}}},
               {"1:5:2,10:15:3", {Slice{1, 5, 2}, Slice{10, 15, 3}}},
               {"1:-1,10:15", {Slice{1, static_cast<uint32_t>(-1), 1}, Slice{10, 15, 1}}},
               {",1:5,,10:15,", {Slice{1, 5, 1}, Slice{10, 15, 1}}},
               {":,:,:",
                {Slice{0, static_cast<uint32_t>(-1), 1},
                 Slice{0, static_cast<uint32_t>(-1), 1},
                 Slice{0, static_cast<uint32_t>(-1), 1}}},
               {"0:,:,:",
                {Slice{0, static_cast<uint32_t>(-1), 1},
                 Slice{0, static_cast<uint32_t>(-1), 1},
                 Slice{0, static_cast<uint32_t>(-1), 1}}}};

        for(auto const& testCase : testCases)
        {
            auto const result = toSlice(testCase.first);
            CHECK_THAT(result, SizeIs(testCase.second.size()));
            CHECK_THAT(result, Catch::Matchers::RangeEquals(testCase.second, SliceEqual{}));
        }
    }


    SECTION("throw on")
    {
        std::vector<std::string> testCases;
        SECTION("non-digit input.")
        {
            testCases = {
                "a:b:c",
                "1:5,7:1:c,10:15",
                "1:5,a:1:99,10:15",
            };
        }
        SECTION("end values < -1.")
        {
            testCases = {
                "1:-5:1,10:15",
            };
        }

        SECTION("-1 values for anything but end value.")
        {
            testCases = {
                "-1:4:1,10:15",
                "1:4:1,10:15:-3",
            };
        }

        SECTION("whitespaces.")
        {
            testCases = {" 1:5", "1: 5", "1:5 "};
        }


        for(auto const& testCase : testCases)
        {
            CHECK_THROWS_AS(toSlice(testCase), std::runtime_error);
        }
    }
}

TEST_CASE("toRangeSlice interprets single int as single slice.")
{
    std::vector<std::pair<std::string, std::vector<Slice>>> testCases = {
        {"5", {Slice{5, 6, 1}}},
        {"5,10", {Slice{5, 6, 1}, Slice{10, 11, 1}}},
    };

    for(auto const& testCase : testCases)
    {
        auto const result = toRangeSlice(testCase.first);
        CHECK_THAT(result, SizeIs(testCase.second.size()));
        CHECK_THAT(result, Catch::Matchers::RangeEquals(testCase.second, SliceEqual{}));
    }
}

TEST_CASE("toTimeSlice interprets single int as step size.")
{
    std::vector<std::pair<std::string, std::vector<Slice>>> testCases = {
        {"5", {Slice{0, static_cast<uint32_t>(-1), 5}}},
        {"5,10", {Slice{0, static_cast<uint32_t>(-1), 5}, Slice{0, static_cast<uint32_t>(-1), 10}}},
    };

    for(auto const& testCase : testCases)
    {
        auto const result = toTimeSlice(testCase.first);
        CHECK_THAT(result, SizeIs(testCase.second.size()));
        CHECK_THAT(result, Catch::Matchers::RangeEquals(testCase.second, SliceEqual{}));
    }
}
