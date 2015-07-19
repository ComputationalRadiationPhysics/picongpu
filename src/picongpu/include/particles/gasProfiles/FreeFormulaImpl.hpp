/**
 * Copyright 2015 Rene Widera, Richard Pausch
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
#include "simulationControl/MovingWindow.hpp"

namespace picongpu
{

namespace gasProfiles
{

template<typename T_ParamClass>
struct FreeFormulaImpl : public T_ParamClass
{
    typedef T_ParamClass ParamClass;

    template<typename T_SpeciesType>
    struct apply
    {
        typedef FreeFormulaImpl<ParamClass> type;
    };

    HINLINE FreeFormulaImpl(uint32_t currentStep)
    {
    }

    /** Calculate the gas density
     *
     * @param totalCellOffset total offset including all slides [in cells]
     */
    HDINLINE float_X operator()(const DataSpace<simDim>& totalCellOffset)
    {
        DataSpace<simDim> position_SI = totalCellOffset;
        cellSize_t cellSize_SI = cellSize;
        for(unsigned int i = 0; i<simDim ; i++)
        {
            cellSize_SI *= UNIT_LENGTH;
            position_SI[i] *= cellSize_SI[i];
        }

        float_X density = ParamClass::operator()(position_SI, cellSize_SI);

        return density;
    }

};
} //namespace gasProfiles
} //namespace picongpu
