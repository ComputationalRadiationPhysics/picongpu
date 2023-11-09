/* Copyright 2023 Tapish Narwal
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

#include "picongpu/param/binningSetup.param"
#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/plugins/binning/BinningCreator.hpp"

#include <cstdint>
#include <memory>
#include <variant>

namespace picongpu
{
    namespace plugins::binning
    {
        /**
         * @brief Master plugin that holds and calls the vector of the actual plugins doing the binning
         */
        class BinningDispatcher : public ISimulationPlugin
        {
        private:
            std::string pluginName;
            std::vector<std::unique_ptr<INotify>> binnerVector;

        public:
            // constructor doesn't know command line arguments if any, use pluginLoad to finish initialization
            BinningDispatcher() : pluginName{"Binning Dispatcher: A flexible binning plugin!"}
            {
                Environment<>::get().PluginConnector().registerPlugin(this);
            }

            ~BinningDispatcher() override = default;

            void pluginRegisterHelp(po::options_description& desc) override
            {
                return;
            }

            std::string pluginGetName() const override
            {
                return pluginName;
            }

            /**
             * Create the binners
             * Set mapping Description for the "dispatched" binners
             */
            void setMappingDescription(MappingDesc* cellDescription) override
            {
                /**
                 * Create and register Binning Plugins
                 */
                BinningCreator binningCreator{binnerVector, cellDescription};
                getBinning(binningCreator);
            }

            void restart(uint32_t restartStep, const std::string restartDirectory) override
            {
                /* restart from a checkpoint here
                 * will be called only once per simulation and before notify() */
            }

            void checkpoint(uint32_t currentStep, const std::string restartDirectory) override
            {
                /* create a persistent checkpoint here
                 * will be called before notify() if both will be called for the same timestep */
            }

            void notify(uint32_t currentStep) override
            {
            }

        private:
            void pluginLoad() override
            {
            }
        };
    } // namespace plugins::binning
} // namespace picongpu
