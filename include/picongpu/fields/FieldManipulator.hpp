/* Copyright 2013-2019 Axel Huebl, Rene Widera
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
#include "FieldManipulator.kernel"
#include "picongpu/simulationControl/MovingWindow.hpp"
#include "picongpu/fields/laserProfiles/profiles.hpp"

#include <pmacc/traits/GetNumWorkers.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/memory/dataTypes/Mask.hpp>

#include <string>
#include <sstream>

namespace picongpu
{

using namespace pmacc;

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

                /* allow to enable the absorber on the top side if the laser
                 * initialization plane in y direction is *not* in cell zero
                 */
                if (fields::laserProfiles::Selected::initPlaneY == 0)
                {
                    /* disable the absorber on top side if
                     *      no slide was performed and
                     *      laser init time is not over
                     */
                    if (numSlides == 0 && ((currentStep * DELTA_T) <= fields::laserProfiles::Selected::INIT_TIME))
                    {
                        /* disable absorber on top side */
                        if (i == TOP) continue;
                    }
                }

                /* if sliding window is active we disable absorber on bottom side*/
                if (MovingWindow::getInstance().isSlidingWindowActive(currentStep) && i == BOTTOM) continue;

                ExchangeMapping<GUARD, MappingDesc> mapper(cellDescription, i);
                constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                    pmacc::math::CT::volume< SuperCellSize >::type::value
                >::value;

                PMACC_KERNEL( KernelAbsorbBorder< numWorkers> {} )(
                    mapper.getGridDim(),
                    numWorkers
                )(
                    deviceBox,
                    thickness,
                    absorber_strength,
                    mapper
                );
            }
        }
    }

    static pmacc::traits::StringProperty getStringProperties()
    {
        pmacc::traits::StringProperty propList;
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

