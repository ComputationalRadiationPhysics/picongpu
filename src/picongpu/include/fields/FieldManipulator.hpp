/**
 * Copyright 2013 Axel Huebl, Rene Widera
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
 


#ifndef FIELDMANIPULATOR_HPP
#define	FIELDMANIPULATOR_HPP

#include "types.h"
#include "fields/LaserPhysics.hpp"
#include "simulation_defines.hpp"
#include "simulation_classTypes.hpp"
#include "mappings/simulation/GridController.hpp"
#include "FieldManipulator.kernel"
#include "simulationControl/MovingWindow.hpp"

namespace picongpu
{

using namespace PMacc;

class FieldManipulator
{
public:

    template<class BoxedMemory>
    static void absorbBorder(uint32_t currentStep, MappingDesc &cellDescription, BoxedMemory deviceBox)
    {
        VirtualWindow win = MovingWindow::getInstance().getVirtualWindow(currentStep);
        for (uint32_t i = 1; i < numberOfNeighbors[simDim]; ++i)
        {
            /* only call for plains: left right top bottom back front*/
            if (FRONT % i == 0 && !(GridController<simDim>::getInstance().getCommunicationMask().isSet(i)))
            {
                uint32_t direction = 0; /*set direction to X (default)*/
                if (i >= BOTTOM && TOP <= TOP)
                    direction = 1; /*set direction to Y*/
                if (i >= BACK)
                    direction = 2; /*set direction to Z*/

                /* exchange mod 2 to find positiv or negitive direction
                 * positiv direction = 1
                 * negativ direction = 0
                 */
                uint32_t pos_or_neg = i % 2;

                uint32_t thickness = ABSORBER_CELLS[direction][pos_or_neg];
                float_X absorber_strength = ABSORBER_STRENGTH[direction][pos_or_neg];

                if (thickness == 0) continue; /*if the absorber has no thickness we check the next side*/

                /* disable the absorber on top side if
                 *      no slide was performed and
                 *      laser init time is not over
                 */
                if (win.slides == 0 && ((currentStep * DELTA_T) <= laserProfile::INIT_TIME))
                {
                    if (i == TOP) continue; /*disable laser on top side*/
                }

                /* if sliding window is active we disable absorber on bottom side*/
                if (MovingWindow::getInstance().isSlidingWindowActive() && i == BOTTOM) continue;

                ExchangeMapping<GUARD, MappingDesc> mapper(cellDescription, i);
                __cudaKernel(kernelAbsorbBorder)
                    (mapper.getGridDim(), mapper.getSuperCellSize())
                    (deviceBox, thickness, absorber_strength,
                     mapper);
            }
        }
    }
};


} //namespace


#endif	/* FIELDMANIPULATOR_HPP */

