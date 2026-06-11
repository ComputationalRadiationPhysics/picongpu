/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/pluginSystem/INotify.hpp>

#include <memory>

namespace picongpu
{
    namespace plugins
    {
        namespace multi
        {
            struct IHelp;

            /** Interface for a single instance of a plugin
             *
             * A plugin which fulfills this interface can be used an instance plugin for multi::Master.
             *
             * An instance must register itself to the PluginConnector to receive the notify calls.
             */
            struct IInstance : public pmacc::INotify
            {
                //! must be implemented by the user
                static std::shared_ptr<IHelp> getHelp();

                //! restart the plugin from a checkpoint
                virtual void restart(uint32_t restartStep, std::string const& restartDirectory) = 0;

                //! create a check point for the plugin
                virtual void checkpoint(uint32_t currentStep, std::string const& checkpointDirectory) = 0;

                /**
                 * Called each timestep if particles are leaving the global simulation volume.
                 *
                 * The order in which the plugins are called is undefined, so this means
                 * read-only access to the particles.
                 *
                 * @param speciesName name of the particle species
                 * @param direction the direction the particles are leaving the simulation
                 */
                virtual void onParticleLeave(std::string const& speciesName, int32_t const direction)
                {
                }
            };

        } // namespace multi
    } // namespace plugins
} // namespace picongpu
