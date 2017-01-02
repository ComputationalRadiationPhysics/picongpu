/**
 * Copyright 2013-2016 Axel Huebl, Rene Widera
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

#include "simulation_defines.hpp"
#include "simulation_classTypes.hpp"
#include "mappings/simulation/GridController.hpp"
#include "memory/dataTypes/Mask.hpp"
#include "FieldManipulator.kernel"
#include "simulationControl/MovingWindow.hpp"

#include <string>
#include <sstream>

namespace picongpu
{

using namespace PMacc;

class FieldManipulator
{
public:

    template<class BoxedMemory>
    static void absorbBorder(uint32_t currentStep, MappingDesc &cellDescription, BoxedMemory deviceBox)
    {
        const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
        for (uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i)
        {
            /* only call for planes: left right top bottom back front*/
            if (FRONT % i == 0 && !(Environment<simDim>::get().GridController().getCommunicationMask().isSet(i)))
            {
                uint32_t direction = 0; /*set direction to X (default)*/
                if (i >= BOTTOM && i <= TOP)
                    direction = 1; /*set direction to Y*/
                if (i >= BACK)
                    direction = 2; /*set direction to Z*/

                /* exchange mod 2 to find positive or negative direction
                 * positive direction = 1
                 * negative direction = 0
                 */
                uint32_t pos_or_neg = i % 2;

                uint32_t thickness = ABSORBER_CELLS[direction][pos_or_neg];
                float_X absorber_strength = ABSORBER_STRENGTH[direction][pos_or_neg];

                if (thickness == 0) continue; /*if the absorber has no thickness we check the next side*/

                /* disable the absorber on top side if
                 *      no slide was performed and
                 *      laser init time is not over
                 */
                if (numSlides == 0 && ((currentStep * DELTA_T) <= laserProfile::INIT_TIME))
                {
                    if (i == TOP) continue; /*disable laser on top side*/
                }

                /* if sliding window is active we disable absorber on bottom side*/
                if (MovingWindow::getInstance().isSlidingWindowActive() && i == BOTTOM) continue;

                ExchangeMapping<GUARD, MappingDesc> mapper(cellDescription, i);
                PMACC_KERNEL(KernelAbsorbBorder{})
                    (mapper.getGridDim(), mapper.getSuperCellSize())
                    (deviceBox, thickness, absorber_strength,
                     mapper);
            }
        }
    }

    static PMacc::traits::StringProperty getStringProperties()
    {
        PMacc::traits::StringProperty propList;
        const DataSpace<DIM3> periodic =
            Environment<simDim>::get().EnvironmentController().getCommunicator().getPeriodic();

        for( uint32_t i = 1; i < NumberOfExchanges<simDim>::value; ++i )
        {
            // for each planar direction: left right top bottom back front
            if( FRONT % i == 0 )
            {
                const std::string directionName = ExchangeTypeNames()[i];
                const DataSpace<DIM3> relDir = Mask::getRelativeDirections<DIM3>(i);

                bool isPeriodic = false;
                uint32_t axis = 0;    // x(0) y(1) z(2)
                uint32_t axisDir = 0; // negative (0), positive (1)
                for( uint32_t d = 0; d < simDim; d++ )
                {
                    if( relDir[d] * periodic[d] != 0 )
                        isPeriodic = true;
                    if( relDir[d] != 0 )
                        axis = d;
                }
                if( relDir[axis] > 0 )
                    axisDir = 1;

                std::string boundaryName = "open"; // absorbing boundary
                if( isPeriodic )
                    boundaryName = "periodic";

                if( boundaryName == "open" )
                {
                    std::ostringstream boundaryParam;
                    boundaryParam << "exponential damping over "
                                  << ABSORBER_CELLS[axis][axisDir] << " cells";
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
};
} //namespace

