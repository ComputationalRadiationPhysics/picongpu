/* Copyright 2013-2023 Axel Huebl, Benjamin Schneider, Felix Schmitt,
 *                     Heiko Burau, Rene Widera, Richard Pausch,
 *                     Benjamin Worpitz, Erik Zenker, Finn-Ole Carstens,
 *                     Franz Poeschel
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


#include "picongpu/plugins/ILightweightPlugin.hpp"
#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/plugins/PluginRegistry.hpp"

#include <pmacc/mappings/kernel/MappingDescription.hpp>

#include <list>
#include <memory>


namespace picongpu
{
    /**
     * Plugin management controller for user-level plugins.
     */
    class PluginController : public ILightweightPlugin
    {
    private:
        std::vector<std::shared_ptr<ISimulationPlugin>> plugins;

        /**
         * Initializes the controller by adding all user plugins to its internal list.
         */
        virtual void init()
        {
            auto const pluginFactories = PluginRegistry::get().getPluginFactories();
            plugins.reserve(pluginFactories.size());
            for(auto const& pluginFactory : pluginFactories)
            {
                plugins.emplace_back(std::move(pluginFactory->createPlugin()));
            }
        }

    public:
        PluginController()
        {
            init();
        }

        ~PluginController() override = default;

        void setMappingDescription(MappingDesc* cellDescription) override
        {
            PMACC_ASSERT(cellDescription != nullptr);

            for(auto plugin : plugins)
            {
                plugin->setMappingDescription(cellDescription);
            }
        }

        void pluginRegisterHelp(po::options_description&) override
        {
            // no help required at the moment
        }

        std::string pluginGetName() const override
        {
            return "PluginController";
        }

        void notify(uint32_t) override
        {
        }

        void pluginUnload() override
        {
            plugins.clear();
        }
    };

} // namespace picongpu
