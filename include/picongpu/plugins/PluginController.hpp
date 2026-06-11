/*
 * SPDX-FileCopyrightText: Axel Huebl, Benjamin Schneider, Felix Schmitt, Heiko Burau, Rene Widera, Richard Pausch, Benjamin Worpitz, Erik Zenker, Finn-Ole Carstens, Franz Poeschel
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/plugins/ILightweightPlugin.hpp"
#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/plugins/PluginRegistry.hpp"

#include <pmacc/mappings/kernel/MappingDescription.hpp>

#include <memory>
#include <vector>

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
            // get all plugins from plugin registry
            auto const pluginFactories = PluginRegistry::get().getPluginFactories();
            plugins.reserve(pluginFactories.size());
            for(auto const& pluginFactory : pluginFactories)
            {
                plugins.emplace_back(pluginFactory->createPlugin());
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
