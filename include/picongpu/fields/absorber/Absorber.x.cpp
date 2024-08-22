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


#include "picongpu/fields/absorber/Absorber.hpp"

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/filter/filter.hpp"

#include <pmacc/Environment.hpp>

#include <memory>
#include <sstream>
#include <type_traits>


namespace picongpu
{
    namespace fields
    {
        namespace absorber
        {
            Absorber::Absorber(Kind const kind) : kind(kind)
            {
                switch(kind)
                {
                case Kind::Exponential:
                    name = std::string{"exponential damping"};
                    break;
                case Kind::None:
                    name = std::string{"none"};
                    break;
                case Kind::Pml:
                    name = std::string{"convolutional PML"};
                    break;
                default:
                    throw std::runtime_error("Unsupported absorber kind requested to be made");
                }
                // Read thickness from absorber.param
                for(uint32_t axis = 0u; axis < simDim; axis++)
                    for(uint32_t direction = 0u; direction < 2u; direction++)
                        numCells[axis][direction] = NUM_CELLS[axis][direction];
            }

            Absorber& Absorber::get()
            {
                // Delay initialization till the first call since the factory has its parameters set during runtime
                static std::unique_ptr<Absorber> pInstance = nullptr;
                if(!pInstance)
                {
                    auto& factory = AbsorberFactory::get();
                    pInstance = factory.make();
                }
                return *pInstance;
            }

            Absorber::Kind Absorber::getKind() const
            {
                return kind;
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
