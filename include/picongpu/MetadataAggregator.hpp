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

#pragma once

#include <pmacc/boost_workaround.hpp>

#include <pmacc/pluginSystem/IPlugin.hpp>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/value_semantic.hpp>

#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>

#include <nlohmann/json.hpp>
#include <picongpu/traits/GetMetadata.hpp>


namespace picongpu
{
    using boost::program_options::options_description;
    using boost::program_options::value;
    using std::ofstream;
    using std::ostream;
    using std::string;
    using std::filesystem::path;


    /** Aggregate metadata to dump
     *
     * This practically wraps its `static json metadata` member and provides functionality to dump it into a json file.
     * It inherits from IPlugin for uniformity with other components but it's not a plugin in the strict sense. To use
     * it, you should basically load(), pluginRegisterHelp(), dump(), unload() (although some of those might be empty).
     */
    struct MetadataAggregator : pmacc::IPlugin
    {
        string pluginGetName() const override
        {
            return "MetadataPlugin";
        }

        void pluginRegisterHelp(options_description& description) override
        {
            description.add_options()(
                "dump-metadata",
                value<path>(&filename)
                    // TODO: One could theoretically use this convention to deactivate explicitly via
                    // `<executable> --dump-metadata ""` which might not quite be the expected behaviour.
                    // We should decide if we want something cleverer here to circumvent this.
                    ->default_value("") // this works like bool_switch -> disable if not given
                    ->implicit_value(defaultFilename) // this provides default value but only if given
                    ->notifier( // this sets `isSupposedToRun`
                        [this](auto const& filename) { this->thisIsSupposedToRun = filename == "" ? false : true; }));
        }

        void notify(uint32_t currentStep) override
        {
        }
        void checkpoint(uint32_t currentStep, const std::string checkpointDirectory) override
        {
        }
        void restart(uint32_t restartStep, const std::string restartDirectory) override
        {
        }
        void dump() const
        {
            if(thisIsSupposedToRun)
            {
                ofstream file{filename};
                dumpTo(file);
            }
        }
        void dumpTo(ostream& stream) const
        {
            stream << metadata.dump(2) << "\n";
        }

        bool thisIsSupposedToRun{false};
        path filename{""};
        const path defaultFilename{
            // doc-include-start: metadata default filename
            "picongpu-metadata.json"
            // doc-include-end: metadata default filename
        };
        inline static nlohmann::json metadata = nlohmann::json::object();
    };

    /** Add metadata to the MetadataAggregator
     *
     * This function takes objects or types the metadata of which should be dumped, extracts the metadata and adds it
     * to the MetadataAggregator. It is important to note that this is not only a registration but immediately extracts
     * the information. So, make sure that all the information you want to add is available at the time you call this
     * function.
     *
     * @tparam T_CT Compiletime type(s) the metadata of which should be added; these are NEVER deduced
     * @tparam T_RT Runtime type(s) the metadata of which should be added; these are ALWAYS deduced and cannot be given
     * explicitly
     * @param obj Runtime instance(s) the metadata of which should be added
     */
    template<typename... T_CT, typename... T_RT>
    void addMetadataOf(T_RT const&... obj)
    {
        // This implementation would theoretically be capable of adding multiple CT and RT metadata sets at once.
        // But it feels like a potential reason for a lot of headache for future-me, so as long as we don't need that
        // feature, we deactivate manually here.
        static_assert(
            sizeof...(T_CT) + sizeof...(T_RT) == 1,
            "As we consider it highly likely to mess up non-trivial cases, please add your metadata one by one.");

        using picongpu::traits::GetMetadata;

        (MetadataAggregator::metadata.merge_patch(GetMetadata<T_CT>{}.description()), ...);
        (MetadataAggregator::metadata.merge_patch(GetMetadata<T_RT>{obj}.description()), ...);
    }

    // the following is only provided to add CT metadata via pmacc::meta::forEach, thus the very restricted interface
    template<typename T>
    struct AddMetadataOf
    {
        void operator()()
        {
            addMetadataOf<T>();
        }
    };

} // namespace picongpu
