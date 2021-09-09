/* Copyright 2021 Sergei Bastrakov
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

#include "picongpu/particles/boundary/ApplyImpl.hpp"
#include "picongpu/particles/boundary/Kind.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/traits/NumberOfExchanges.hpp>

#include <cstdint>
#include <stdexcept>


namespace picongpu
{
    namespace particles
    {
        namespace boundary
        {
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
                auto const numExchanges = NumberOfExchanges<simDim>::value;
                auto const communicationMask = Environment<simDim>::get().GridController().getCommunicationMask();
                for(uint32_t exchange = 1u; exchange < numExchanges; ++exchange)
                {
                    /* Here we are only interested in the positive and negative
                     * directions for x, y, z axes and not the "diagonal" ones.
                     * So skip other directions except left, right, top, bottom,
                     * back, front
                     */
                    if(FRONT % exchange != 0)
                        continue;

                    // If this is not an outer boundary, also skip
                    bool hasNeighbour = communicationMask.isSet(exchange);
                    if(hasNeighbour)
                        continue;

                    // Transform exchange into axis
                    uint32_t axis = 0;
                    if(exchange >= BOTTOM && exchange <= TOP)
                        axis = 1;
                    if(exchange >= BACK)
                        axis = 2;

                    auto boundaryKind = species.boundaryKind()[axis];
                    switch(boundaryKind)
                    {
                    case Kind::Periodic:
                        ApplyImpl<Kind::Periodic>{}(species, exchange, currentStep);
                        break;
                    case Kind::Absorbing:
                        ApplyImpl<Kind::Absorbing>{}(species, exchange, currentStep);
                        break;
                    default:
                        throw std::runtime_error("Unsupported boundary kind when trying to apply particle boundary");
                    }
                }
            }

        } // namespace boundary
    } // namespace particles
} // namespace picongpu
