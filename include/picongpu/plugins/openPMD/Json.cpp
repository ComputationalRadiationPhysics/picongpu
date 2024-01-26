/* Copyright 2021-2023 Franz Poeschel
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#if ENABLE_OPENPMD == 1

#    include "picongpu/plugins/openPMD/Json.hpp"

#    include "picongpu/plugins/common/openPMDVersion.def"
#    include "picongpu/plugins/openPMD/Json_private.hpp"

#    include <algorithm> // std::copy_n, std::find
#    include <cctype> // std::isspace

#    include <openPMD/auxiliary/JSON.hpp>
#    include <openPMD/openPMD.hpp>

/*
 * Note:
 * This is a hostonly .cpp file because CMake will not use -isystem for system
 * include paths on NVCC targets created with cupla_add_executable.
 * Since <nlohmann/json.hpp> throws a number of warnings, this .cpp file
 * ensures that NVCC never sees that library.
 */

namespace picongpu
{
    namespace json
    {
        void MatcherPerBackend::init(nlohmann::json const& config)
        {
            if(config.is_object())
            {
                // simple layout: only one global JSON object was passed
                // forward this one directly to openPMD
                m_patterns.emplace_back("", std::make_shared<nlohmann::json>(config));
                m_defaultConfig = config;
            }
            else if(config.is_array())
            {
                bool defaultEmplaced = false;
                // enhanced PIConGPU-defined layout
                for(size_t i = 0; i < config.size(); ++i)
                {
                    auto kindOfConfig = readPattern(m_patterns, m_defaultConfig, config[i]);
                    if(kindOfConfig == KindOfConfig::Default)
                    {
                        if(defaultEmplaced)
                        {
                            throw std::runtime_error(
                                "[openPMD plugin] Specified more than one default configuration.");
                        }
                        else
                        {
                            defaultEmplaced = true;
                        }
                    }
                }
            }
            else
            {
                throw std::runtime_error("[openPMD plugin] Expecting an object or an array as JSON configuration.");
            }
        }

        /**
         * @brief Get the JSON config associated with a regex pattern.
         *
         * @param datasetPath The regex.
         * @return The matched JSON configuration, as a string.
         */
        nlohmann::json const& MatcherPerBackend::get(std::string const& datasetPath) const
        {
            for(auto const& pattern : m_patterns)
            {
                if(std::regex_match(datasetPath, pattern.pattern))
                {
                    return *pattern.config;
                }
            }
            static nlohmann::json const emptyConfig; // null
            return emptyConfig;
        }

        void JsonMatcher::init(std::string const& config, MPI_Comm comm)
        {
            auto const filename = extractFilename(config);
            m_wholeConfig = nlohmann::json::parse(filename.empty() ? config : collective_file_read(filename, comm));
            if(!m_wholeConfig.is_object())
            {
                throw std::runtime_error("[openPMD plugin] Expected an object for the JSON configuration.");
            }
            m_perBackend.reserve(m_wholeConfig.size());
            for(auto it = m_wholeConfig.begin(); it != m_wholeConfig.end(); ++it)
            {
                std::string const& backendName = it.key();
                if(std::find(m_recognizedBackends.begin(), m_recognizedBackends.end(), backendName)
                   == m_recognizedBackends.end())
                {
                    // The key does not point to the configuration of a backend recognized by PIConGPU
                    // Ignore it.
                    continue;
                }
                if(!it.value().is_object())
                {
                    throw std::runtime_error(
                        "[openPMD plugin] Each backend's configuration must be a JSON object (config for backend "
                        + backendName + ").");
                }
                if(it.value().contains("dataset"))
                {
                    m_perBackend.emplace_back(PerBackend{backendName, MatcherPerBackend{it.value().at("dataset")}});
                }
            }
        }
        std::string JsonMatcher::get(std::string const& datasetPath) const
        {
            nlohmann::json result = nlohmann::json::object();
            for(auto const& backend : m_perBackend)
            {
                auto const& datasetConfig = backend.matcher.get(datasetPath);
                if(datasetConfig.empty())
                {
                    // ensure that there actually is an object to erase this from
                    result[backend.backendName]["dataset"] = {};
                    result[backend.backendName].erase("dataset");
                }
                else
                {
                    result[backend.backendName]["dataset"] = datasetConfig;
                }
            }
            return result.dump();
        }

        std::string JsonMatcher::getDefault() const
        {
            nlohmann::json result = m_wholeConfig;
            for(auto const& backend : m_perBackend)
            {
                auto const& datasetConfig = backend.matcher.getDefault();
                if(datasetConfig.empty())
                {
                    // ensure that there actually is an object to erase this from
                    result[backend.backendName]["dataset"] = {};
                    result[backend.backendName].erase("dataset");
                }
                else
                {
                    result[backend.backendName]["dataset"] = datasetConfig;
                }
            }
            // note that at this point, config[<backend>][dataset] is no longer
            // a list, the list has been resolved by the previous loop
            addDefaults(result);
            return result.dump();
        }

        std::unique_ptr<AbstractJsonMatcher> AbstractJsonMatcher::construct(std::string const& config, MPI_Comm comm)
        {
            return std::unique_ptr<AbstractJsonMatcher>{new JsonMatcher{config, comm}};
        }
    } // namespace json
} // namespace picongpu

// Anonymous namespace so these helpers don't get exported
namespace
{
    /**
     * @brief Remove leading and trailing characters from a string.
     *
     * @tparam F Functor type for to_remove
     * @param s String to trim.
     * @param to_remove Functor deciding which characters to remove.
     */
    template<typename F>
    std::string trim(std::string const& s, F&& to_remove)
    {
        auto begin = s.begin();
        for(; begin != s.end(); ++begin)
        {
            if(!to_remove(*begin))
            {
                break;
            }
        }
        auto end = s.rbegin();
        for(; end != s.rend(); ++end)
        {
            if(!to_remove(*end))
            {
                break;
            }
        }
        return s.substr(begin - s.begin(), end.base() - begin);
    }

    /**
     * @brief Check whether the string points to a filename or not.
     *
     * A string is considered to point to a filename if its first
     * non-whitespace character is an '@'.
     * The filename will be trimmed of whitespace using trim().
     *
     * @param unparsed The string that possibly points to a file.
     * @return The filename if the string points to the file, an empty
     *         string otherwise.
     *
     * @todo Upon switching to C++17, use std::optional to make the return
     *       type clearer.
     *       Until then, this is somewhat safe anyway since filenames need
     *       to be non-empty.
     */
    std::string extractFilename(std::string const& unparsed)
    {
        std::string trimmed = trim(unparsed, [](char c) { return std::isspace(c); });
        if(trimmed.at(0) == '@')
        {
            trimmed = trimmed.substr(1);
            trimmed = trim(trimmed, [](char c) { return std::isspace(c); });
            return trimmed;
        }
        else
        {
            return {};
        }
    }

    KindOfConfig readPattern(
        std::vector<picongpu::json::Pattern>& patterns,
        nlohmann::json& defaultConfig,
        nlohmann::json const& object)
    {
        static std::string const errorMsg = R"END(
[openPMD plugin] Each single pattern in an extended JSON configuration
must be a JSON object with keys 'select' and 'cfg'.
The key 'select' is optional, indicating a default configuration if it is
not set.
The key 'select' must point to either a single string or an array of strings.)END";

        if(!object.is_object())
        {
            throw std::runtime_error(errorMsg);
        }
        try
        {
            nlohmann::json const& cfg = object.at("cfg");
            if(!object.contains("select"))
            {
                nlohmann::json const& cfg = object.at("cfg");
                defaultConfig = cfg;
                return KindOfConfig::Default;
            }
            else
            {
                nlohmann::json const& pattern = object.at("select");
                auto cfgShared = std::make_shared<nlohmann::json>(cfg);
                if(pattern.is_string())
                {
                    patterns.emplace_back(pattern.get<std::string>(), std::move(cfgShared));
                }
                else if(pattern.is_array())
                {
                    patterns.reserve(pattern.size());
                    for(size_t i = 0; i < pattern.size(); ++i)
                    {
                        patterns.emplace_back(pattern[i].get<std::string>(), cfgShared);
                    }
                }
                else
                {
                    throw std::runtime_error(errorMsg);
                }
                return KindOfConfig::Pattern;
            }
        }
        catch(nlohmann::json::out_of_range const&)
        {
            throw std::runtime_error(errorMsg);
        }
    }

    void addDefaults(nlohmann::json& config)
    {
        /*
         * hdf5.dataset.chunks = none
         * --------------------------
         * Disable HDF5 chunking as it can conflict with MPI-IO backends.
         * This is very likely the same issue as
         * https://github.com/open-mpi/ompi/issues/7795.
         *
         *
         * adios2.engine.preferred_flush_target = "buffer"
         * -----------------------------------------------
         * Only relevant for ADIOS2 engines that support this feature,
         * ignored otherwise. Currently supported in BP5.
         * Small datasets should be written to the internal ADIOS2
         * buffer.
         * Big datasets should explicitly specify their flush target
         * in Series::flush(). Options are "buffer" and "disk".
         * Ideally, all flush() calls should specify this explicitly.
         *
         *
         * adios2.engine.parameters.BufferChunkSize = 2147381248
         * -----------------------------------------------------
         * This parameter is only interpreted by the ADIOS2 BP5 engine
         * (and potentially future engines that use the BP5 serializer).
         * Other engines will ignore it without warning.
         *
         * Reasoning: The internal data structure of BP5 is a linked
         * list of equally-sized chunks.
         * This parameter specifies the size of each individual chunk to
         * the maximum possible 2GB (i.e. a bit lower than that),
         * which is more performant than the default 128MB.
         *
         * Since each buffer chunk is allocated by malloc(), chunks are
         * not actually written upon creation.
         * As a result, on systems with virtual memory, the overhead
         * of specifying this is a potential allocation of up to 2GB
         * of unused **virtual** memory, **not physical** memory.
         * That's a good deal, since it gives us performance by default.
         *
         * On those systems where this setting actually poses a problem,
         * careful memory configuration is necessary anyway, so the
         * defaults don't matter.
         */
        std::string const& defaultValues = R"(
{
  "hdf5": {
    "dataset": {
      "chunks": "none"
    }
  },
  "adios2": {
    "engine": {
      "preferred_flush_target": "buffer",
      "parameters": {
        "BufferChunkSize": 2147381248
      }
    }
  }
}
        )";
        std::stringstream json_to_string;
        json_to_string << config;
        auto merged = openPMD::json::merge(defaultValues, json_to_string.str());
        config = nlohmann::json::parse(merged);
    }
} // namespace

#endif // ENABLE_OPENPMD
