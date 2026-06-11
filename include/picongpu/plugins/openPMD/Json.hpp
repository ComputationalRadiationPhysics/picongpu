/*
 * SPDX-FileCopyrightText: Franz Poeschel
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <memory> // std::unique_ptr
#include <string>

#include <mpi.h>

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
        class AbstractJsonMatcher
        {
        public:
            /**
             * @brief Construct a JSON matcher to hand out dataset-specific configurations
             *
             * This function will parse the given config, after reading it
             * from a file if needed. In this case, the constructor is
             * MPI-collective.
             * It will distinguish per backend between ordinary openPMD JSON configurations
             * and extended configurations as defined by PIConGPU.
             * If an ordinary JSON configuration was detected, given regex
             * patterns will be matched against "" (the empty string).
             *
             * @param config The JSON configuration, exactly as in --openPMD.backendConfig.
             * @param comm MPI communicator for collective file reading, if needed.
             * @return std::unique_ptr<AbstractJsonMatcher>
             */
            static std::unique_ptr<AbstractJsonMatcher> construct(std::string const& config, MPI_Comm comm);

            virtual ~AbstractJsonMatcher() = default;

            /**
             * @brief Get the JSON config associated with a regex pattern.
             *
             * @param datasetPath The regex.
             * @return The matched JSON configuration, as a string.
             */
            virtual std::string get(std::string const& datasetPath) const = 0;

            /**
             * @brief Get the default JSON config.
             *
             * @return The default JSON configuration, as a string.
             */
            virtual std::string getDefault() const = 0;
        };
    } // namespace json
} // namespace picongpu
