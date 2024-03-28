/* Copyright 2017-2023 Rene Widera
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

#include "picongpu/simulation_defines.hpp"

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
                virtual void onParticleLeave(const std::string& speciesName, const int32_t direction)
                {
                }
            };

        } // namespace multi
    } // namespace plugins
} // namespace picongpu
