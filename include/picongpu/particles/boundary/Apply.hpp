/* Copyright 2021-2023 Sergei Bastrakov
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

#include "picongpu/particles/boundary/Absorbing.hpp"
#include "picongpu/particles/boundary/ApplyImpl.hpp"
#include "picongpu/particles/boundary/Description.hpp"
#include "picongpu/particles/boundary/Reflecting.hpp"
#include "picongpu/particles/boundary/Thermal.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/boundary/Utility.hpp>
#include <pmacc/pluginSystem/IPlugin.hpp>

#include <cstdint>
#include <list>
#include <stdexcept>


namespace picongpu
{
    namespace particles
    {
        namespace boundary
        {
            namespace detail
            {
                /** Call onParticleLeave() hooks for all plugins for the given outer boundary
                 *
                 * It is only called for active outer boundaries.
                 * Hook implementations must not modify the particles.
                 * Hook implementations must account for actual boundary positions set in
                 * T_Species::boundaryDescription() and not assume it is in GUARD.
                 *
                 * @tparam T_Species particle species type
                 *
                 * @param exchangeType exchange describing the active boundary
                 */
                template<typename T_Species>
                inline void callPluginHooks(T_Species& species, int32_t const exchangeType)
                {
                    using Plugins = std::list<pmacc::IPlugin*>;
                    Plugins plugins = Environment<>::get().PluginConnector().getAllPlugins();
                    for(Plugins::iterator iter = plugins.begin(); iter != plugins.end(); iter++)
                        (*iter)->onParticleLeave(T_Species::FrameType::getName(), exchangeType);
                }
            } // namespace detail

            /** Apply boundary conditions to the given species
             *
             * @tparam T_Species particle species type
             *
             * @param species particle species
             * @param currentStep current time iteration
             */
            template<typename T_Species>
            inline void apply(T_Species&& species, uint32_t currentStep)
            {
                auto const communicationMask = Environment<simDim>::get().GridController().getCommunicationMask();
                for(auto exchange : getAllAxisAlignedExchanges())
                {
                    // If this is not an outer boundary, also skip
                    bool hasNeighbour = communicationMask.isSet(exchange);
                    if(hasNeighbour)
                        continue;

                    /* Here is the only place in the computational loop where we can call hooks from plugins:
                     *    - we know those particles crossed the active boundary
                     *    - but we didn't yet apply the boundary conditions
                     *      that would normally modify or delete the particles
                     *      (this is done just afterwards in this function)
                     */
                    detail::callPluginHooks(species, exchange);

                    // Call boundary condition implementation
                    auto axis = pmacc::boundary::getAxis(exchange);
                    auto boundaryDescription = species.boundaryDescription()[axis];
                    switch(boundaryDescription.kind)
                    {
                    case Kind::Periodic:
                        ApplyImpl<Kind::Periodic>{}(species, exchange, currentStep);
                        break;
                    case Kind::Absorbing:
                        ApplyImpl<Kind::Absorbing>{}(species, exchange, currentStep);
                        break;
                    case Kind::Reflecting:
                        ApplyImpl<Kind::Reflecting>{}(species, exchange, currentStep);
                        break;
                    case Kind::Thermal:
                        ApplyImpl<Kind::Thermal>{}(species, exchange, currentStep);
                        break;
                    default:
                        throw std::runtime_error("Unsupported boundary kind when trying to apply particle boundary");
                    }
                }
            }

        } // namespace boundary
    } // namespace particles
} // namespace picongpu
