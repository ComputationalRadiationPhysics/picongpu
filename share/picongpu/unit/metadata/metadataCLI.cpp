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
#include "picongpu/MetadataAggregator.hpp"

#include <boost/program_options/options_description.hpp>

#include <filesystem>
#include <string>

#include <catch2/catch_test_macros.hpp>

using boost::program_options::options_description;
using picongpu::ArgsParser;
using picongpu::MetadataAggregator;
using std::string;
using std::vector;
using std::filesystem::path;


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

    size_t makeArgc() const
    {
        return content.size();
    };
};

struct TestableArgsParser : ArgsParser
{
    static TestableArgsParser& getInstance()
    {
        static TestableArgsParser instance;
        return instance;
    }
    void reset()
    {
        options.clear();
    }
};

TEST_CASE("unit::metadataCLI", "[metadata CLI test]")
{
    MetadataAggregator metadataAggregator;
    TestableArgsParser& argsParser = TestableArgsParser::getInstance();
    argsParser.reset();

    options_description description(metadataAggregator.pluginGetName());
    metadataAggregator.pluginRegisterHelp(description);
    argsParser.addOptions(description);

    SECTION("deactivated by default")
    {
        FictitiousArgv fictitiousArgv{{"<executable>"}};
        argsParser.parse(fictitiousArgv.makeArgc(), fictitiousArgv.makeArgv());
        CHECK(!metadataAggregator.thisIsSupposedToRun);
    }

    SECTION("gets activated via `--dump-metadata`")
    {
        FictitiousArgv fictitiousArgv{{"<executable>", "--dump-metadata"}};
        argsParser.parse(fictitiousArgv.makeArgc(), fictitiousArgv.makeArgv());
        CHECK(metadataAggregator.thisIsSupposedToRun);
    }

    SECTION("has correct default filename")
    {
        FictitiousArgv fictitiousArgv{{"<executable>", "--dump-metadata"}};
        argsParser.parse(fictitiousArgv.makeArgc(), fictitiousArgv.makeArgv());
        CHECK(metadataAggregator.filename == metadataAggregator.defaultFilename);
    }

    SECTION("gets activated with additional filename")
    {
        string filename{"filename"};
        FictitiousArgv fictitiousArgv{{"<executable>", "--dump-metadata", filename}};
        argsParser.parse(fictitiousArgv.makeArgc(), fictitiousArgv.makeArgv());
        CHECK(metadataAggregator.thisIsSupposedToRun);
    }

    SECTION("takes filename after `--dump-metadata`")
    {
        string filename{"filename"};
        FictitiousArgv fictitiousArgv{{"<executable>", "--dump-metadata", filename}};
        argsParser.parse(fictitiousArgv.makeArgc(), fictitiousArgv.makeArgv());
        CHECK(metadataAggregator.filename == filename);
    }
}
