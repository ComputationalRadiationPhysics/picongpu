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

#pragma once

#include "picongpu/plugins/openPMD/Json.hpp"

#include <mpi.h>
#include <nlohmann/json.hpp>

#include <regex>
#include <string>
#include <vector>

/*
 * Note:
 * This header is included only into hostonly .cpp files because CMake
 * will not use -isystem for system include paths on NVCC targets created
 * with cupla_add_executable.
 * Since <nlohmann/json.hpp> throws a number of warnings, this design
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
    std::string trim(std::string const& s, F&& to_remove);

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
    std::string extractFilename(std::string const& unparsed);

    /**
     * @brief Helper class to help figure out a platform-independent
     *        MPI_Datatype for size_t.
     */
    template<typename>
    struct MPI_Types;

    template<>
    struct MPI_Types<unsigned long>
    {
        // can't make this constexpr due to MPI
        // so, make this non-static for simplicity
        MPI_Datatype value = MPI_UNSIGNED_LONG;
    };

    template<>
    struct MPI_Types<unsigned long long>
    {
        MPI_Datatype value = MPI_UNSIGNED_LONG_LONG;
    };

    template<>
    struct MPI_Types<unsigned>
    {
        MPI_Datatype value = MPI_UNSIGNED;
    };

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
    std::string collective_file_read(std::string const& path, MPI_Comm comm);

    struct Pattern
    {
        std::regex pattern;
        std::shared_ptr<nlohmann::json const> config;

        Pattern(std::string pattern_in, std::shared_ptr<nlohmann::json const> config_in)
            // we construct the patterns once and use them often, so let's ask for some optimization
            : pattern{std::move(pattern_in), std::regex_constants::egrep | std::regex_constants::optimize}
            , config{std::move(config_in)}
        {
        }
    };

    enum class KindOfConfig : char
    {
        Pattern,
        Default
    };

    /**
     * @brief Read a single JSON pattern of the form {"select": ..., "cfg": ...}
     *
     * The "select" key is optional, indicating the default configuration if it
     * is missing.
     *
     * @param patterns Output parameter: Emplace a parsed pattern into this list.
     * @param defaultConfig Output parameter: If the pattern was the default pattern,
     *                      emplace it here.
     * @param object The JSON object that is parsed as the pattern.
     * @return Whether the pattern was the default configuration or not.
     */
    KindOfConfig readPattern(
        std::vector<Pattern>& patterns,
        nlohmann::json& defaultConfig,
        nlohmann::json const& object);

    /**
     * @brief Matcher for dataset configurations per backend.
     *
     */
    class MatcherPerBackend
    {
    private:
        nlohmann::json m_defaultConfig;
        std::vector<Pattern> m_patterns;

        void init(nlohmann::json const& config);

    public:
        /**
         * @brief For default construction.
         */
        explicit MatcherPerBackend() = default;

        /**
         * @brief Initialize one backend's JSON matcher from its configuration.
         *
         * This constructor will parse the given config.
         * It will distinguish between ordinary openPMD JSON configurations
         * and extended configurations as defined by PIConGPU.
         * If an ordinary JSON configuration was detected, given regex
         * patterns will be matched against "" (the empty string).
         *
         * @param config The JSON configuration for one backend.
         *               E.g. for ADIOS2, this will be the sub-object/array found under
         *               config["adios2"]["dataset"].
         */
        MatcherPerBackend(nlohmann::json const& config)
        {
            init(config);
        }

        /**
         * @brief Get the JSON config associated with a regex pattern.
         *
         * @param datasetPath The regex.
         * @return The matched JSON configuration, as a string.
         */
        nlohmann::json const& get(std::string const& datasetPath) const;

        /**
         * @brief Get the default JSON config.
         *
         * @return The default JSON configuration, as a string.
         */
        nlohmann::json const& getDefault() const
        {
            return m_defaultConfig;
        }
    };
} // namespace

namespace picongpu
{
    namespace json
    {
        /**
         * @brief Class to handle extended JSON configurations as used by
         *        the openPMD plugin.
         *
         * This class handles parsing of the extended JSON patterns as well as
         * selection of one JSON configuration by regex.
         *
         */
        class JsonMatcher : public AbstractJsonMatcher
        {
        private:
            struct PerBackend
            {
                std::string backendName;
                MatcherPerBackend matcher;
            };
            std::vector<PerBackend> m_perBackend;
            nlohmann::json m_wholeConfig;
            static std::vector<std::string> const m_recognizedBackends;

            void init(std::string const& config, MPI_Comm comm);

        public:
            /**
             * @brief For default construction.
             */
            explicit JsonMatcher() = default;

            /**
             * @brief Initialize JSON matcher from command line arguments.
             *
             * This constructor will parse the given config, after reading it
             * from a file if needed. In this case, the constructor is
             * MPI-collective.
             * It will distinguish between ordinary openPMD JSON configurations
             * and extended configurations as defined by PIConGPU.
             * If an ordinary JSON configuration was detected, given regex
             * patterns will be matched against "" (the empty string).
             *
             * @param config The JSON configuration, exactly as in
             *               --openPMD.json.
             * @param comm MPI communicator for collective file reading,
             *             if needed.
             */
            JsonMatcher(std::string const& config, MPI_Comm comm)
            {
                init(config, comm);
            }

            /**
             * @brief Get the JSON config associated with a regex pattern.
             *
             * @param datasetPath The regex.
             * @return The matched JSON configuration, as a string.
             */
            std::string get(std::string const& datasetPath) const override;

            /**
             * @brief Get the default JSON config.
             *
             * @return The default JSON configuration, as a string.
             */
            std::string getDefault() const override;
        };

        std::vector<std::string> const JsonMatcher::m_recognizedBackends = {"adios1", "adios2", "hdf5", "json"};
    } // namespace json
} // namespace picongpu
