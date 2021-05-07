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

#include "picongpu/fields/absorber/Absorber.hpp"
#include "picongpu/fields/absorber/ExponentialDamping.hpp"
#include "picongpu/fields/absorber/Pml.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/traits/IsBaseTemplateOf.hpp>

#include <sstream>
#include <type_traits>


namespace picongpu
{
    namespace fields
    {
        namespace maxwellSolver
        {
            /** Forward declaration to avoid mutual including with YeePML.hpp
             *
             * @tparam T_CurlE functor to compute curl of E
             * @tparam T_CurlB functor to compute curl of B
             */
            template<typename T_CurlE, typename T_CurlB>
            class YeePML;

        } // namespace maxwellSolver

        namespace absorber
        {
            // This implementation has to go to a .tpp file as it requires definitions of Pml and ExponentialDamping
            Absorber& Absorber::get()
            {
                // This is currently a static type, until absorbers go full runtime
                constexpr bool isPmlSolver
                    = pmacc::traits::IsBaseTemplateOf_t<maxwellSolver::YeePML, fields::Solver>::value;
                using Implementation = std::conditional_t<isPmlSolver, Pml, ExponentialDamping>;
                static Implementation instance;
                return instance;
            }

            Thickness Absorber::getGlobalThickness() const
            {
                Thickness thickness;
                for(uint32_t axis = 0u; axis < 3u; axis++)
                    for(uint32_t direction = 0u; direction < 2u; direction++)
                        thickness(axis, direction) = numCells[axis][direction];
                const DataSpace<DIM3> isPeriodicBoundary
                    = Environment<simDim>::get().EnvironmentController().getCommunicator().getPeriodic();
                for(uint32_t axis = 0u; axis < 3u; axis++)
                    if(isPeriodicBoundary[axis])
                    {
                        thickness(axis, 0) = 0u;
                        thickness(axis, 1) = 0u;
                    }
                return thickness;
            }

            Thickness Absorber::getLocalThickness() const
            {
                Thickness thickness = getGlobalThickness();
                auto const numExchanges = NumberOfExchanges<simDim>::value;
                auto const communicationMask = Environment<simDim>::get().GridController().getCommunicationMask();
                for(uint32_t exchange = 1u; exchange < numExchanges; exchange++)
                {
                    /* Here we are only interested in the positive and negative
                     * directions for x, y, z axes and not the "diagonal" ones.
                     * So skip other directions except left, right, top, bottom,
                     * back, front
                     */
                    if(FRONT % exchange != 0)
                        continue;

                    // Transform exchange into a pair of axis and direction
                    uint32_t axis = 0;
                    if(exchange >= BOTTOM && exchange <= TOP)
                        axis = 1;
                    if(exchange >= BACK)
                        axis = 2;
                    uint32_t direction = exchange % 2;

                    // No absorber at the borders between two local domains
                    bool hasNeighbour = communicationMask.isSet(exchange);
                    if(hasNeighbour)
                        thickness(axis, direction) = 0u;
                }
                return thickness;
            }

            pmacc::traits::StringProperty Absorber::getStringProperties()
            {
                auto& absorber = get();
                auto const thickness = absorber.getGlobalThickness();
                pmacc::traits::StringProperty propList;
                const DataSpace<DIM3> periodic
                    = Environment<simDim>::get().EnvironmentController().getCommunicator().getPeriodic();

                for(uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i)
                {
                    // for each planar direction: left right top bottom back front
                    if(FRONT % i == 0)
                    {
                        const std::string directionName = ExchangeTypeNames()[i];
                        const DataSpace<DIM3> relDir = Mask::getRelativeDirections<DIM3>(i);

                        bool isPeriodic = false;
                        uint32_t axis = 0; // x(0) y(1) z(2)
                        uint32_t axisDir = 0; // negative (0), positive (1)
                        for(uint32_t d = 0; d < simDim; d++)
                        {
                            if(relDir[d] * periodic[d] != 0)
                                isPeriodic = true;
                            if(relDir[d] != 0)
                                axis = d;
                        }
                        if(relDir[axis] > 0)
                            axisDir = 1;

                        std::string boundaryName = "open"; // absorbing boundary
                        if(isPeriodic)
                            boundaryName = "periodic";

                        if(boundaryName == "open")
                        {
                            std::ostringstream boundaryParam;
                            boundaryParam << absorber.name + " over " << thickness(axis, axisDir) << " cells";
                            propList[directionName]["param"] = boundaryParam.str();
                        }
                        else
                        {
                            propList[directionName]["param"] = "none";
                        }

                        propList[directionName]["name"] = boundaryName;
                    }
                }
                return propList;
            }

        } // namespace absorber
    } // namespace fields
} // namespace picongpu
