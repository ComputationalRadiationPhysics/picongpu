/* Copyright 2021 Franz Poeschel
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
#    include "picongpu/plugins/openPMD/Json_private.hpp"

#    include <algorithm> // std::copy_n, std::find
#    include <cctype> // std::isspace
#    include <fstream>
#    include <sstream>

/*
 * Note:
 * This is a hostonly .cpp file because CMake will not use -isystem for system
 * include paths on NVCC targets created with cupla_add_executable.
 * Since <nlohmann/json.hpp> throws a number of warnings, this .cpp file
 * ensures that NVCC never sees that library.
 */

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

    /**
     * @brief Read a file in MPI-collective manner.
     *
     * The file is read on rank 0 and its contents subsequently distributed
     * to all other ranks.
     *
     * @param path Path for the file to read.
     * @param comm MPI communicator.
     * @return std::string Full file content.
     */
    std::string collective_file_read(std::string const& path, MPI_Comm comm)
    {
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        std::string res;
        size_t stringLength = 0;
        if(rank == 0)
        {
            std::fstream handle;
            handle.open(path, std::ios_base::in);
            std::stringstream stream;
            stream << handle.rdbuf();
            res = stream.str();
            if(!handle.good())
            {
                throw std::runtime_error("Failed reading JSON config from file " + path + ".");
            }
            stringLength = res.size() + 1;
        }
        MPI_Datatype datatype = MPI_Types<size_t>{}.value;
        int err = MPI_Bcast(&stringLength, 1, datatype, 0, comm);
        if(err)
        {
            throw std::runtime_error("[collective_file_read] MPI_Bcast stringLength failure.");
        }
        std::vector<char> recvbuf(stringLength, 0);
        if(rank == 0)
        {
            std::copy_n(res.c_str(), stringLength, recvbuf.data());
        }
        err = MPI_Bcast(recvbuf.data(), stringLength, MPI_CHAR, 0, comm);
        if(err)
        {
            throw std::runtime_error("[collective_file_read] MPI_Bcast file content failure.");
        }
        if(rank != 0)
        {
            res = recvbuf.data();
        }
        return res;
    }

    KindOfConfig readPattern(
        std::vector<Pattern>& patterns,
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

    void MatcherPerBackend::init(nlohmann::json const& config)
    {
        if(config.is_object())
        {
            // simple layout: only one global JSON object was passed
            // forward this one directly to openPMD
            m_patterns.emplace_back("", std::make_shared<nlohmann::json>(config));
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
                        throw std::runtime_error("[openPMD plugin] Specified more than one default configuration.");
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
} // namespace

namespace picongpu
{
    namespace json
    {
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
                    continue;
                }
                result[backend.backendName]["dataset"] = datasetConfig;
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
                    continue;
                }
                result[backend.backendName]["dataset"] = datasetConfig;
            }
            return result.dump();
        }

        std::unique_ptr<AbstractJsonMatcher> AbstractJsonMatcher::construct(std::string const& config, MPI_Comm comm)
        {
            return std::unique_ptr<AbstractJsonMatcher>{new JsonMatcher{config, comm}};
        }
    } // namespace json
} // namespace picongpu

#endif // ENABLE_OPENPMD
