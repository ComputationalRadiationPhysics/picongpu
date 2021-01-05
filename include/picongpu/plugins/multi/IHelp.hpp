/* Copyright 2017-2021 Rene Widera
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

#include "picongpu/simulation_defines.hpp"
#include "picongpu/plugins/multi/ISlave.hpp"


namespace picongpu
{
    namespace plugins
    {
        namespace multi
        {
            //! Interface to expose a help of a plugin
            struct IHelp
            {
                //! creates a ISlave instance
                virtual std::shared_ptr<ISlave> create(
                    std::shared_ptr<IHelp>& help,
                    size_t const id,
                    MappingDesc* cellDescription)
                    = 0;

                /** register help options
                 *
                 * The options are used if the plugin is a ISlave and is handling
                 * there own notification period.
                 */
                virtual void registerHelp(
                    boost::program_options::options_description& desc,
                    std::string const& masterPrefix = std::string{})
                    = 0;

                /** register independent help options
                 *
                 * This options can be used even if the plugin is not handling there
                 * own notification period.
                 */
                virtual void expandHelp(
                    boost::program_options::options_description& desc,
                    std::string const& masterPrefix = std::string{})
                    = 0;

                //! validate if the command line interface options are well formated
                virtual void validateOptions() = 0;

                //! number of plugin which must be created
                virtual size_t getNumPlugins() const = 0;

                //! short description of the plugin functionality
                virtual std::string getDescription() const = 0;

                //! name of the plugin
                virtual std::string getName() const = 0;
            };

        } // namespace multi
    } // namespace plugins
} // namespace picongpu
