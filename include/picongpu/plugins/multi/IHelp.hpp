/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/plugins/multi/IInstance.hpp"

namespace picongpu
{
    namespace plugins
    {
        namespace multi
        {
            //! Interface to expose a help of a plugin
            struct IHelp
            {
                //! creates an instance
                virtual std::shared_ptr<IInstance> create(
                    std::shared_ptr<IHelp>& help,
                    size_t const id,
                    MappingDesc* cellDescription)
                    = 0;

                /** register help options
                 *
                 * The options are used if the plugin is a IInstance and is handling
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
