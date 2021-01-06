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
#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/plugins/multi/ISlave.hpp"
#include "picongpu/plugins/multi/IHelp.hpp"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"

#include <vector>
#include <list>
#include <stdexcept>


namespace picongpu
{
    namespace plugins
    {
        namespace multi
        {
            /** Master class to create multi plugins
             *
             * Create and handle a plugin as multi plugin. Parameter of a multi plugin
             * can be used multiple times on the command line.
             *
             * @tparam T_Slave type of the plugin (must inherit from ISlave)
             */
            template<typename T_Slave>
            class Master : public ISimulationPlugin
            {
            public:
                using Slave = T_Slave;
                using SlaveList = std::list<std::shared_ptr<ISlave>>;
                SlaveList slaveList;

                std::shared_ptr<IHelp> slaveHelp;

                MappingDesc* m_cellDescription = nullptr;

                Master() : slaveHelp(Slave::getHelp())
                {
                    Environment<>::get().PluginConnector().registerPlugin(this);
                }

                virtual ~Master()
                {
                }

                std::string pluginGetName() const
                {
                    // the PMacc plugin system needs a short description instead of the plugin name
                    return slaveHelp->getName() + ": " + slaveHelp->getDescription();
                }

                void pluginRegisterHelp(boost::program_options::options_description& desc)
                {
                    slaveHelp->registerHelp(desc);
                }

                void setMappingDescription(MappingDesc* cellDescription)
                {
                    m_cellDescription = cellDescription;
                }

                /** restart a checkpoint
                 *
                 * Trigger the method restart() for all slave instances.
                 */
                void restart(uint32_t restartStep, std::string const restartDirectory)
                {
                    for(auto& slave : slaveList)
                        slave->restart(restartStep, restartDirectory);
                }

                /** create a checkpoint
                 *
                 * Trigger the method checkpoint() for all slave instances.
                 */
                void checkpoint(uint32_t currentStep, std::string const checkpointDirectory)
                {
                    for(auto& slave : slaveList)
                        slave->checkpoint(currentStep, checkpointDirectory);
                }

            private:
                void pluginLoad()
                {
                    size_t const numSlaves = slaveHelp->getNumPlugins();
                    if(numSlaves > 0u)
                        slaveHelp->validateOptions();
                    for(size_t i = 0; i < numSlaves; ++i)
                    {
                        slaveList.emplace_back(slaveHelp->create(slaveHelp, i, m_cellDescription));
                    }
                }

                void pluginUnload()
                {
                    slaveList.clear();
                }

                void notify(uint32_t currentStep)
                {
                    // nothing to do here
                }
            };

        } // namespace multi
    } // namespace plugins

    namespace particles
    {
        namespace traits
        {
            template<typename T_Species, typename T_Slave>
            struct SpeciesEligibleForSolver<T_Species, plugins::multi::Master<T_Slave>>
            {
                using type = typename SpeciesEligibleForSolver<T_Species, T_Slave>::type;
            };
        } // namespace traits
    } // namespace particles
} // namespace picongpu
