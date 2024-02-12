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

#include "picongpu/ArgsParser.hpp"

#include <boost/program_options/options_description.hpp>

#include <catch2/catch_test_macros.hpp>

using boost::program_options::options_description;
using picongpu::ArgsParser;
using std::string;
using std::vector;

struct MetadataPlugin
{
    string pluginGetName()
    {
        return "MetadataPlugin";
    }

    void pluginRegisterHelp(options_description& description)
    {
    }

    bool isSupposedToRun{false};
};

// strongly inspired by
// https://codereview.stackexchange.com/questions/205269/create-a-c-style-char-from-a-c-vectorstring/205298#205298
// under CC-BY-SA 4.0
struct FictitiousArgv
{
    vector<string> content;
    vector<char*> pointers;

    char** makeArgv()
    {
        pointers.resize(content.size() + 1);
        transform(begin(content), end(content), begin(pointers), [](auto& in) { return in.data(); });
        pointers[content.size()] = nullptr;
        return pointers.data();
    }

    size_t size() const
    {
        return content.size();
    };
};

TEST_CASE("unit::metadataCLI", "[metadata CLI test]")
{
    MetadataPlugin metadataPlugin;
    ArgsParser& ap = ArgsParser::getInstance();
    options_description description(metadataPlugin.pluginGetName());
    metadataPlugin.pluginRegisterHelp(description);
    ap.addOptions(description);


    SECTION("deactivated by default")
    {
        FictitiousArgv fictitiousArgv{{"<executable>"}};
        ap.parse(fictitiousArgv.size(), fictitiousArgv.makeArgv());
        CHECK(!metadataPlugin.isSupposedToRun);
    }

    SECTION("gets activated via `--dump-metadata`")
    {
        FictitiousArgv fictitiousArgv{{"<executable>", "--dump-metadata"}};
        ap.parse(fictitiousArgv.size(), fictitiousArgv.makeArgv());
        CHECK(metadataPlugin.isSupposedToRun);
    }
}
