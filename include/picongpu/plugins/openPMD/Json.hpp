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

#include <mpi.h>

#include <memory> // std::unique_ptr
#include <string>

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
             * @param config The JSON configuration, exactly as in --openPMD.json.
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
