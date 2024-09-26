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

#include "picongpu/defines.hpp"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/plugins/multi/IHelp.hpp"
#include "picongpu/plugins/multi/IInstance.hpp"

#include <list>
#include <stdexcept>
#include <vector>


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
             * @tparam T_Instance (single) instance type of the plugin, a child class of IInstance
             */
            template<typename T_Instance>
            class Master : public ISimulationPlugin
            {
            public:
                using Instance = T_Instance;
                using InstanceList = std::list<std::shared_ptr<IInstance>>;
                InstanceList instanceList;

                std::shared_ptr<IHelp> instanceHelp;

                MappingDesc* m_cellDescription = nullptr;

                Master() : instanceHelp(Instance::getHelp())
                {
                    Environment<>::get().PluginConnector().registerPlugin(this);
                }

                ~Master() override = default;

                std::string pluginGetName() const override
                {
                    // the PMacc plugin system needs a short description instead of the plugin name
                    return instanceHelp->getName() + ": " + instanceHelp->getDescription();
                }

                void pluginRegisterHelp(boost::program_options::options_description& desc) override
                {
                    instanceHelp->registerHelp(desc);
                }

                void setMappingDescription(MappingDesc* cellDescription) override
                {
                    m_cellDescription = cellDescription;
                }

                /** restart a checkpoint
                 *
                 * Trigger the method restart() for all instances.
                 */
                void restart(uint32_t restartStep, std::string const restartDirectory) override
                {
                    for(auto& instance : instanceList)
                        instance->restart(restartStep, restartDirectory);
                }

                /** Call the onParticleLeave() method for all instances.
                 *
                 * Called each timestep if particles are leaving the global simulation volume.
                 *
                 * @param speciesName name of the particle species
                 * @param direction the direction the particles are leaving the simulation
                 */
                void onParticleLeave(const std::string& speciesName, const int32_t direction) override
                {
                    for(auto& instance : instanceList)
                        instance->onParticleLeave(speciesName, direction);
                }

                /** create a checkpoint
                 *
                 * Trigger the method checkpoint() for all instances.
                 */
                void checkpoint(uint32_t currentStep, std::string const checkpointDirectory) override
                {
                    for(auto& instance : instanceList)
                        instance->checkpoint(currentStep, checkpointDirectory);
                }

            private:
                void pluginLoad() override
                {
                    size_t const numInstances = instanceHelp->getNumPlugins();
                    if(numInstances > 0u)
                        instanceHelp->validateOptions();
                    for(size_t i = 0; i < numInstances; ++i)
                    {
                        instanceList.emplace_back(instanceHelp->create(instanceHelp, i, m_cellDescription));
                    }
                }

                void pluginUnload() override
                {
                    instanceList.clear();
                }

                void notify(uint32_t currentStep) override
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
            template<typename T_Species, typename T_Instance>
            struct SpeciesEligibleForSolver<T_Species, plugins::multi::Master<T_Instance>>
            {
                using type = typename SpeciesEligibleForSolver<T_Species, T_Instance>::type;
            };
        } // namespace traits
    } // namespace particles
} // namespace picongpu
